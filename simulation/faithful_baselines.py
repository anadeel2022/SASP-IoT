"""Explicit single-hop scheduling objectives, with exact joint assignment.

All methods see the same estimated CSI, HOL packet, queue, battery and power
limits as SASP. No method accesses future arrivals or latent physical CSI.
The finite-buffer/deadline simulator is not a setting in which classical
infinite-horizon stability theorems can simply be asserted.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from channel_env import TRAFFIC_CLASSES
from sasp_scheduler import _hungarian_max

METHODS = ("MaxRateMatching", "MaxWeight", "WeightedMaxWeight", "EDF", "LyapunovDPP-Energy")
BIT_REFERENCE = 1000.0
ENERGY_REFERENCE_J = 1e-3
INVALID = -1e100


@dataclass(frozen=True)
class BaselineConfig:
    name: str
    dpp_v: float = 1.0

    def __post_init__(self):
        if self.name not in METHODS:
            raise ValueError(f"Unknown baseline: {self.name}")
        if not np.isfinite(self.dpp_v) or self.dpp_v < 0:
            raise ValueError("DPP V must be finite and nonnegative")


def edge_utility(name, backlog_bits, service_bits, energy_j, priority=1.0, dpp_v=1.0):
    """Dimensionless objective; scaling does not alter MaxWeight assignments."""
    service = service_bits / BIT_REFERENCE
    if name in ("MaxRateMatching", "EDF"):
        return service
    score = (backlog_bits / BIT_REFERENCE) * service
    if name == "WeightedMaxWeight":
        return priority * score
    if name == "LyapunovDPP-Energy":
        return score - dpp_v * energy_j / ENERGY_REFERENCE_J
    if name == "MaxWeight":
        return score
    raise ValueError(name)


def build_pair_table(env, cfg):
    """Maximize over allowed powers for each edge before exact matching.

    Power choices are independent across edges: there is no shared power
    budget in this environment. This elimination is therefore exact.
    Equal power scores choose the lowest power index, without perturbing the
    primary objective. Zero/nonpositive edges can be replaced by idle.
    """
    env.prepare_slot()
    backlog = [sum(p.remaining_bits for p in q) for q in env.queues]
    score = np.full((env.n_channels, env.n_devices), INVALID, dtype=float)
    power = np.full(score.shape, -1, dtype=int)
    predictions = {}
    for ch in range(env.n_channels):
        mask = env.valid_action_mask(ch, np.zeros(env.n_devices, dtype=bool))
        for dev in range(env.n_devices):
            priority = TRAFFIC_CLASSES[int(env.device_class[dev])].priority
            for pidx in range(env.n_power):
                if not mask[env.action_index(dev, pidx)]:
                    continue
                prediction = env.predict_action_service(dev, ch, pidx)
                bits = prediction["predicted_service_bits"]
                energy = prediction["predicted_energy_j"]
                # Zero service is weakly dominated by idle for every objective.
                if bits <= 0 or energy > env.battery_j[dev] + 1e-12:
                    continue
                utility = edge_utility(cfg.name, backlog[dev], bits, energy, priority, cfg.dpp_v)
                if utility > score[ch, dev]:
                    score[ch, dev] = utility
                    power[ch, dev] = pidx
                    predictions[ch, dev] = prediction
    if cfg.name == "EDF":
        # HOL-only EDF adaptation to parallel channels. Select the C earliest
        # absolute deadlines, then maximize service within that selected set.
        eligible = [d for d in range(env.n_devices) if np.any(power[:, d] >= 0)]
        eligible.sort(key=lambda d: (env.queues[d][0].deadline_slot,
                                     env.queues[d][0].arrival_slot, d))
        chosen = set(eligible[:env.n_channels])
        for dev in eligible:
            if dev not in chosen:
                score[:, dev] = INVALID
                power[:, dev] = -1
    return score, power, predictions


def select_baseline_actions(env, cfg):
    scores, powers, predictions = build_pair_table(env, cfg)
    # One dummy column per channel permits any subset of channels to idle.
    matrix = np.concatenate((scores, np.zeros((env.n_channels, env.n_channels))), axis=1)
    columns = _hungarian_max(matrix)
    actions = np.zeros(env.n_channels, dtype=int)
    telemetry = []
    for ch, dev in enumerate(columns):
        item = {"channel": ch, "device": -1, "power_index": -1,
                "objective": 0.0, "predicted_service_bits": 0.0, "predicted_energy_j": 0.0}
        if dev < env.n_devices and powers[ch, dev] >= 0 and scores[ch, dev] > 0:
            pidx = int(powers[ch, dev])
            actions[ch] = env.action_index(dev, pidx)
            item.update(predictions[ch, dev])
            item.update(device=int(dev), power_index=pidx, objective=float(scores[ch, dev]))
        item["action"] = int(actions[ch])
        telemetry.append(item)
    return actions, telemetry
