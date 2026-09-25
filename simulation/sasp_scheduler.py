"""Actor-free service-aware spectrum scheduler for SASP v8.9.

This module intentionally depends only on NumPy and the environment.  It does
not import PyTorch and cannot load a neural checkpoint.  All persistent state
used by SASP is a four-dimensional virtual queue that is reset to zero at the
start of each scenario and updated causally after observing a slot.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from channel_env import ChannelAgentSpectrumEnv, TRAFFIC_CLASSES


DEFAULT_TARGETS: Tuple[float, float, float, float] = (0.04, 0.05, 0.20, 0.03)


@dataclass(frozen=True)
class SASPConfig:
    targets: Tuple[float, float, float, float] = DEFAULT_TARGETS
    virtual_queue_enabled: bool = True
    virtual_queue_step_size: float = 1.00
    virtual_queue_decay: float = 0.010
    virtual_queue_max: float = 30.0
    service_drain_coef: float = 1.45
    service_completion_coef: float = 0.55
    service_backpressure_coef: float = 1.10
    service_energy_coef: float = 0.08
    critical_coverage_enabled: bool = True
    use_predicted_service: bool = True
    assignment_solver: str = "hungarian"
    queue_margin: int = 2
    emergency_slack: int = 3
    medical_slack: int = 8
    best_effort_slack: int = 18

    def __post_init__(self) -> None:
        if self.assignment_solver not in {"hungarian", "greedy"}:
            raise ValueError("assignment_solver must be 'hungarian' or 'greedy'.")
        if len(self.targets) != 4:
            raise ValueError("targets must contain three deadline targets and one overflow target")


def _virtual_queue_step(
    virtual_queue: np.ndarray,
    arrivals: np.ndarray,
    expired: np.ndarray,
    overflow: np.ndarray,
    cfg: SASPConfig,
) -> np.ndarray:
    current = np.asarray(virtual_queue, dtype=float)
    arrivals = np.asarray(arrivals, dtype=float)
    expired = np.asarray(expired, dtype=float)
    overflow = np.asarray(overflow, dtype=float)
    targets = np.asarray(cfg.targets, dtype=float)
    increment = np.zeros(4, dtype=float)
    increment[:3] = expired[:3] - targets[:3] * arrivals[:3]
    increment[3] = float(np.sum(overflow)) - targets[3] * float(np.sum(arrivals))
    updated = (1.0 - float(cfg.virtual_queue_decay)) * current
    updated += float(cfg.virtual_queue_step_size) * increment
    return np.clip(updated, 0.0, float(cfg.virtual_queue_max))


def _critical_devices(env: ChannelAgentSpectrumEnv, cfg: SASPConfig) -> list[dict[str, float]]:
    limits = (cfg.emergency_slack, cfg.medical_slack, cfg.best_effort_slack)
    capacity = max(int(env.cfg.max_queue_packets), 1)
    result: list[dict[str, float]] = []
    for dev, queue in enumerate(env.queues):
        if not queue or env.battery_j[dev] <= 0.0:
            continue
        packet = queue[0]
        cls = int(packet.class_id)
        slack = int(packet.deadline_slot - env.t)
        near_full = len(queue) >= max(capacity - max(int(cfg.queue_margin), 0), 1)
        urgent = slack <= int(limits[cls])
        if near_full or urgent:
            result.append({
                "device": float(dev),
                "class_id": float(cls),
                "queue_len": float(len(queue)),
                "slack_slots": float(slack),
            })
    return result


def _action_components(
    env: ChannelAgentSpectrumEnv,
    channel: int,
    action: int,
    virtual_queue: np.ndarray,
    cfg: SASPConfig,
) -> dict[str, float]:
    queues = [len(queue) for queue in env.queues]
    global_q = float(np.mean(queues)) / max(int(env.cfg.max_queue_packets), 1)
    vq = np.asarray(virtual_queue, dtype=float)
    vq_n = vq / (1.0 + vq)
    if int(action) == 0:
        max_pressure = 0.0
        for queue in env.queues:
            if not queue:
                continue
            packet = queue[0]
            profile = TRAFFIC_CLASSES[int(packet.class_id)]
            slack_frac = (packet.deadline_slot - env.t) / max(profile.deadline_slots, 1)
            urgency = float(np.clip(1.0 - slack_frac, 0.0, 1.5))
            q_frac = float(len(queue) / max(env.cfg.max_queue_packets, 1))
            max_pressure = max(max_pressure, urgency + q_frac)
        queue_component = float(-0.90 * global_q)
        urgency_component = float(-0.62 * max_pressure)
        virtual_component = float(
            -0.80 * vq_n[3] * global_q - 0.30 * np.max(vq_n[:3]) * max_pressure
        )
        score = queue_component + urgency_component + virtual_component
        return {
            "score": score,
            "policy_component": 0.0,
            "checkpoint_dual_component": 0.0,
            "virtual_component": virtual_component,
            "urgency_component": urgency_component,
            "queue_component": queue_component,
            "cqi_component": 0.0,
            "service_drain_component": 0.0,
            "service_completion_component": 0.0,
            "critical_service_component": 0.0,
            "energy_component": 0.0,
            "predicted_service_bits": 0.0,
            "predicted_packet_completion": 0.0,
            "predicted_head_bits": 0.0,
            "predicted_sinr_db": -100.0,
            "predicted_energy_j": 0.0,
            "final_device": -1.0,
            "final_power_index": -1.0,
            "critical_candidate": 0.0,
            "predicted_service_scoring_enabled": float(cfg.use_predicted_service),
        }

    dev, pidx = env.decode_action(int(action))
    if dev < 0 or dev >= env.n_devices or not env.queues[dev]:
        return {"score": -1e9}
    packet = env.queues[dev][0]
    cls = int(packet.class_id)
    profile = TRAFFIC_CLASSES[cls]
    q_len = len(env.queues[dev])
    capacity = max(int(env.cfg.max_queue_packets), 1)
    q_frac = float(np.clip(q_len / capacity, 0.0, 1.0))
    slack_slots = int(packet.deadline_slot - env.t)
    urgency = float(np.clip(1.0 - slack_slots / max(profile.deadline_slots, 1), 0.0, 1.5))
    slack_limits = (cfg.emergency_slack, cfg.medical_slack, cfg.best_effort_slack)
    critical = float(
        q_len >= max(capacity - max(int(cfg.queue_margin), 0), 1)
        or slack_slots <= int(slack_limits[cls])
    )
    prediction = env.predict_action_service(int(dev), int(channel), int(pidx))
    prediction_weight = float(cfg.use_predicted_service)
    completion = prediction_weight * float(prediction["predicted_packet_completion"])
    service_norm = prediction_weight * float(np.clip(
        prediction["predicted_service_bits"] / max(float(profile.payload_bits), 1.0),
        0.0,
        1.5,
    ))
    cqi = prediction_weight * float(np.clip(
        (float(prediction["predicted_sinr_db"]) + 5.0) / 30.0,
        0.0,
        1.0,
    ))
    priority = float(profile.priority / 3.0)
    power_frac = float(pidx / max(env.n_power - 1, 1))
    battery_frac = float(env.battery_j[dev] / max(profile.battery_j, 1e-12))
    class_pressure = float(vq_n[cls]) * priority * (0.90 * urgency + 0.80 * q_frac)
    overflow_pressure = float(vq_n[3]) * (1.25 * q_frac + 0.55 * urgency + 0.20 * global_q)
    service_drain_component = float(
        cfg.service_drain_coef * service_norm
        * (0.55 + cfg.service_backpressure_coef * (0.55 * q_frac + 0.45 * urgency))
    )
    service_completion_component = float(
        cfg.service_completion_coef * completion * (0.70 + 0.30 * priority)
    )
    critical_service_component = float(
        0.45 * critical * completion * (0.55 + 0.45 * q_frac)
    )
    urgency_component = float(0.62 * priority * urgency)
    queue_component = float(0.66 * q_frac + 0.16 * global_q)
    cqi_component = float(0.14 * cqi)
    virtual_component = float(1.12 * class_pressure + 1.28 * overflow_pressure)
    energy_component = float(
        -cfg.service_energy_coef * power_frac * max(0.0, 0.60 - battery_frac)
    )
    score = (
        urgency_component + queue_component + cqi_component + virtual_component
        + service_drain_component + service_completion_component
        + critical_service_component + energy_component
    )
    return {
        "score": float(score),
        "policy_component": 0.0,
        "checkpoint_dual_component": 0.0,
        "virtual_component": virtual_component,
        "urgency_component": urgency_component,
        "queue_component": queue_component,
        "cqi_component": cqi_component,
        "service_drain_component": service_drain_component,
        "service_completion_component": service_completion_component,
        "critical_service_component": critical_service_component,
        "energy_component": energy_component,
        "predicted_service_bits": float(prediction["predicted_service_bits"]),
        "predicted_packet_completion": float(prediction["predicted_packet_completion"]),
        "predicted_head_bits": float(prediction["predicted_head_bits"]),
        "predicted_sinr_db": float(prediction["predicted_sinr_db"]),
        "predicted_energy_j": float(prediction["predicted_energy_j"]),
        "final_device": float(dev),
        "final_power_index": float(pidx),
        "critical_candidate": critical,
        "predicted_service_scoring_enabled": prediction_weight,
    }


def _hungarian_max(score: np.ndarray) -> List[int]:
    """Solve a rectangular maximum-weight assignment for rows <= columns."""
    values = np.asarray(score, dtype=float)
    n, m = values.shape
    if n == 0:
        return []
    if n > m:
        raise ValueError("Hungarian implementation requires rows <= columns")
    finite = values[np.isfinite(values)]
    max_value = float(np.max(finite)) if finite.size else 0.0
    cost = np.where(np.isfinite(values), max_value - values, max_value + 1e8)
    u = np.zeros(n + 1, dtype=float)
    v = np.zeros(m + 1, dtype=float)
    p = np.zeros(m + 1, dtype=int)
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf, dtype=float)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [-1] * n
    for j in range(1, m + 1):
        if p[j] > 0:
            assignment[p[j] - 1] = j - 1
    return assignment


def _greedy_max(score: np.ndarray) -> List[int]:
    values = np.asarray(score, dtype=float)
    used: set[int] = set()
    assignment: List[int] = []
    for row in range(values.shape[0]):
        available = [col for col in range(values.shape[1]) if col not in used]
        col = max(available, key=lambda index: float(values[row, index]))
        assignment.append(int(col))
        used.add(int(col))
    return assignment


def select_sasp_actions(
    env: ChannelAgentSpectrumEnv,
    virtual_queue: np.ndarray,
    cfg: SASPConfig,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Score all feasible pairs and return a feasible per-channel schedule."""
    env.prepare_slot()
    zero_reservation = np.zeros(env.n_devices, dtype=bool)
    channels, devices = env.n_channels, env.n_devices
    pair_actions = np.zeros((channels, devices), dtype=int)
    pair_scores = np.full((channels, devices), -1e9, dtype=float)
    pair_details: list[list[Optional[dict[str, float]]]] = [
        [None for _ in range(devices)] for _ in range(channels)
    ]
    idle_scores = np.full(channels, -1e9, dtype=float)
    idle_details: list[dict[str, float]] = []
    for channel in range(channels):
        valid = env.valid_action_mask(channel, zero_reservation)
        idle = _action_components(env, channel, 0, virtual_queue, cfg)
        idle_scores[channel] = float(idle["score"])
        idle_details.append(idle)
        for dev in range(devices):
            candidates = [
                env.action_index(dev, pidx)
                for pidx in range(env.n_power)
                if valid[env.action_index(dev, pidx)]
            ]
            if not candidates:
                continue
            details = [_action_components(env, channel, action, virtual_queue, cfg) for action in candidates]
            best_index = int(np.argmax([item["score"] for item in details]))
            pair_actions[channel, dev] = int(candidates[best_index])
            pair_scores[channel, dev] = float(details[best_index]["score"])
            pair_details[channel][dev] = details[best_index]

    matrix = np.full((channels, devices + channels), -1e9, dtype=float)
    matrix[:, :devices] = pair_scores
    for channel in range(channels):
        matrix[channel, devices:] = idle_scores[channel]
    columns = _hungarian_max(matrix) if cfg.assignment_solver == "hungarian" else _greedy_max(matrix)
    actions = np.zeros(channels, dtype=int)
    selected: list[dict[str, float]] = []
    for channel, column in enumerate(columns):
        if 0 <= column < devices and pair_scores[channel, column] > -1e8:
            actions[channel] = int(pair_actions[channel, column])
            selected.append(dict(pair_details[channel][column] or {}))
        else:
            selected.append(dict(idle_details[channel]))

    critical = _critical_devices(env, cfg)
    critical_ids = {int(item["device"]) for item in critical}
    selected_ids = {
        env.decode_action(int(action))[0]
        for action in actions
        if env.decode_action(int(action))[0] >= 0
    }
    coverage_applied = False
    if cfg.critical_coverage_enabled and critical_ids and not (selected_ids & critical_ids):
        candidates: list[tuple[float, int, int]] = []
        for channel in range(channels):
            for dev in critical_ids:
                if pair_scores[channel, dev] > -1e8:
                    candidates.append((float(pair_scores[channel, dev]), channel, dev))
        if candidates:
            _, channel, dev = max(candidates, key=lambda item: item[0])
            actions[channel] = int(pair_actions[channel, dev])
            selected[channel] = dict(pair_details[channel][dev] or {})
            coverage_applied = True

    critical_backlog = float(sum(item["queue_len"] for item in critical))
    for channel, detail in enumerate(selected):
        dev, pidx = env.decode_action(int(actions[channel]))
        detail.update({
            "final_action": float(actions[channel]),
            "final_device": float(dev),
            "final_power_index": float(pidx),
            "all_feasible_scored": 1.0,
            "service_aware_assignment_enabled": 1.0,
            "joint_assignment": float(cfg.assignment_solver == "hungarian"),
            "assignment_solver_hungarian": float(cfg.assignment_solver == "hungarian"),
            "assignment_solver_greedy": float(cfg.assignment_solver == "greedy"),
            "critical_coverage_enabled": float(cfg.critical_coverage_enabled),
            "critical_coverage_applied": float(coverage_applied),
            "safety_critical_candidate_count": float(len(critical_ids)),
            "safety_critical_backlog_before": critical_backlog,
        })
    return actions, selected


def evaluate_sasp(
    env_factory: Callable[[int], ChannelAgentSpectrumEnv],
    scenario_seeds: Sequence[int],
    cfg: SASPConfig,
    *,
    collect_action_trace: bool = False,
) -> list[dict[str, float]] | tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Evaluate SASP on fixed scenarios with zero state carried across episodes."""
    rows: list[dict[str, float]] = []
    traces: list[dict[str, float]] = []
    phases = ("pre_surge", "surge", "degradation", "recovery", "stationary")
    for episode, scenario in enumerate(scenario_seeds):
        env = env_factory(int(scenario))
        env.reset(seed=int(scenario))
        virtual_queue = np.zeros(4, dtype=float)
        vq_trace: list[np.ndarray] = []
        phase_vq: dict[str, list[np.ndarray]] = {phase: [] for phase in phases}
        updates = 0
        slot_index = 0
        coverage_slots = 0
        while not env.done:
            env.prepare_slot()
            in_measurement = bool(env.t < env.measurement_steps)
            phase = str(env.current_phase)
            virtual_before = virtual_queue.copy()
            actions, telemetry = select_sasp_actions(env, virtual_queue, cfg)
            _, _, _, info = env.step(actions)
            arrivals = np.asarray(info["arrivals_new_by_class"], dtype=float)
            expired = np.asarray(info["deadline_expired_by_class"], dtype=float)
            overflow = np.asarray(info["overflow_new_by_class"], dtype=float)
            if cfg.virtual_queue_enabled:
                virtual_queue = _virtual_queue_step(virtual_queue, arrivals, expired, overflow, cfg)
                updates += 1
            if in_measurement:
                phase_vq.setdefault(phase, []).append(virtual_queue.copy())
                coverage_slots += int(any(item.get("critical_coverage_applied", 0.0) > 0.5 for item in telemetry))
            vq_trace.append(virtual_queue.copy())
            if collect_action_trace and in_measurement:
                service = np.asarray(info.get("service_bits_by_channel", np.zeros(env.n_channels)), dtype=float)
                completed = np.asarray(info.get("packet_completion_by_channel", np.zeros(env.n_channels)), dtype=float)
                for channel, detail in enumerate(telemetry):
                    traces.append({
                        "episode": float(episode),
                        "episode_seed": float(scenario),
                        "slot": float(slot_index),
                        "channel": float(channel),
                        "phase": phase,
                        "final_action": float(detail.get("final_action", 0.0)),
                        "final_device": float(detail.get("final_device", -1.0)),
                        "final_power_index": float(detail.get("final_power_index", -1.0)),
                        "all_feasible_scored": 1.0,
                        "service_aware_assignment_enabled": 1.0,
                        "joint_assignment": float(detail.get("joint_assignment", 0.0)),
                        "assignment_solver_hungarian": float(detail.get("assignment_solver_hungarian", 0.0)),
                        "assignment_solver_greedy": float(detail.get("assignment_solver_greedy", 0.0)),
                        "predicted_service_scoring_enabled": float(detail.get("predicted_service_scoring_enabled", 0.0)),
                        "critical_coverage_enabled": float(detail.get("critical_coverage_enabled", 0.0)),
                        "critical_coverage_applied": float(detail.get("critical_coverage_applied", 0.0)),
                        "policy_component": 0.0,
                        "checkpoint_dual_score_contribution": 0.0,
                        "virtual_queue_score_contribution": float(detail.get("virtual_component", 0.0)),
                        "urgency_component": float(detail.get("urgency_component", 0.0)),
                        "queue_component": float(detail.get("queue_component", 0.0)),
                        "cqi_component": float(detail.get("cqi_component", 0.0)),
                        "service_drain_score_contribution": float(detail.get("service_drain_component", 0.0)),
                        "service_completion_score_contribution": float(detail.get("service_completion_component", 0.0)),
                        "critical_service_score_contribution": float(detail.get("critical_service_component", 0.0)),
                        "energy_score_contribution": float(detail.get("energy_component", 0.0)),
                        "final_score": float(detail.get("score", 0.0)),
                        "predicted_service_bits": float(detail.get("predicted_service_bits", 0.0)),
                        "predicted_packet_completion": float(detail.get("predicted_packet_completion", 0.0)),
                        "predicted_head_bits": float(detail.get("predicted_head_bits", 0.0)),
                        "predicted_sinr_db": float(detail.get("predicted_sinr_db", -100.0)),
                        "predicted_energy_j": float(detail.get("predicted_energy_j", 0.0)),
                        "realized_service_bits": float(service[channel]),
                        "realized_packet_completion": float(completed[channel]),
                        "service_prediction_error_bits": float(service[channel] - detail.get("predicted_service_bits", 0.0)),
                        "virtual_queue_emergency_before": float(virtual_before[0]),
                        "virtual_queue_medical_before": float(virtual_before[1]),
                        "virtual_queue_best_effort_before": float(virtual_before[2]),
                        "virtual_queue_overflow_before": float(virtual_before[3]),
                        "virtual_queue_emergency_after": float(virtual_queue[0]),
                        "virtual_queue_medical_after": float(virtual_queue[1]),
                        "virtual_queue_best_effort_after": float(virtual_queue[2]),
                        "virtual_queue_overflow_after": float(virtual_queue[3]),
                    })
            slot_index += 1

        row = env.metrics()
        mean_vq = np.mean(np.asarray(vq_trace), axis=0) if vq_trace else virtual_queue
        row.update({
            "episode": float(episode),
            "episode_seed": float(scenario),
            "topology_seed": float(scenario),
            "execution_mode_code": 8.9,
            "online_dual_enabled": 0.0,
            "online_dual_updates": 0.0,
            "online_lambda_emergency_mean": 0.0,
            "online_lambda_medical_mean": 0.0,
            "online_lambda_best_effort_mean": 0.0,
            "online_lambda_overflow_mean": 0.0,
            "online_virtual_queue_enabled": float(cfg.virtual_queue_enabled),
            "online_virtual_queue_updates": float(updates),
            "virtual_queue_emergency_mean": float(mean_vq[0]),
            "virtual_queue_medical_mean": float(mean_vq[1]),
            "virtual_queue_best_effort_mean": float(mean_vq[2]),
            "virtual_queue_overflow_mean": float(mean_vq[3]),
            "virtual_queue_emergency_final": float(virtual_queue[0]),
            "virtual_queue_medical_final": float(virtual_queue[1]),
            "virtual_queue_best_effort_final": float(virtual_queue[2]),
            "virtual_queue_overflow_final": float(virtual_queue[3]),
            "safety_shield_enabled": float(cfg.critical_coverage_enabled),
            "safety_shield_slots": float(coverage_slots),
        })
        for phase, values in phase_vq.items():
            if values:
                array = np.asarray(values, dtype=float)
                row[f"phase_{phase}_virtual_queue_class0_mean"] = float(np.mean(array[:, 0]))
                row[f"phase_{phase}_virtual_queue_class1_mean"] = float(np.mean(array[:, 1]))
                row[f"phase_{phase}_virtual_queue_class2_mean"] = float(np.mean(array[:, 2]))
                row[f"phase_{phase}_virtual_queue_overflow_mean"] = float(np.mean(array[:, 3]))
        rows.append(row)
    return (rows, traces) if collect_action_trace else rows
