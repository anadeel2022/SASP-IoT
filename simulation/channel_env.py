"""Spectrum-scheduling environment for actor-free SASP v8.9 and comparators.

Each orthogonal channel/resource block receives idle or a (device, power) pair.
A reservation mask enforces device uniqueness. Learned comparators may use
sequential channel agents; SASP scores synchronized candidates and solves a
joint assignment.

Metric semantics are deliberately explicit:
* queue_overflow_packet_probability: offered packets rejected because the
  destination queue was full;
* deadline_expiration_probability: admitted packets that expire before service
  completes, divided by offered packets in the monitored arrival cohort;
* total_packet_failure_probability: overflow plus deadline expiration, divided
  by offered packets in the monitored arrival cohort;
* queue_saturation_device_slot_probability: fraction of device-slot boundaries
  at which a queue was full immediately before the next arrival process;
* no generic ``blocking_prob`` is emitted because this simulator has no separate
  admission-control policy.

Slot and deadline convention
----------------------------
An arrival stamped ``a`` with deadline ``D`` may be served in slots
``a, ..., a + D - 1``. It expires before scheduling in slot ``a + D``. The
primary measurement window is followed by ``max(D)`` slots so that every
packet offered during the primary window has an observed terminal outcome.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class TrafficClass:
    name: str
    arrival_pps: float
    payload_bits: int
    deadline_slots: int
    priority: float
    battery_j: float
    max_power_dbm: float
    energy_per_bit_j: float


TRAFFIC_CLASSES: Tuple[TrafficClass, ...] = (
    TrafficClass("Emergency", 28.0, 900, 14, 3.0, 30.0, 20.0, 1.0e-8),
    TrafficClass("Medical", 12.0, 1600, 65, 1.8, 24.0, 15.0, 0.9e-8),
    TrafficClass("BestEffort", 7.0, 2600, 180, 1.0, 18.0, 10.0, 1.7e-8),
)


@dataclass
class Packet:
    remaining_bits: float
    arrival_slot: int
    deadline_slot: int
    class_id: int
    arrival_phase: str
    metric_eligible: bool = True
    warm_start: bool = False


@dataclass
class ChannelEnvConfig:
    n_devices: int = 12
    n_channels: int = 3
    steps_per_episode: int = 100
    slot_s: float = 1e-3
    channel_bandwidth_hz: float = 180_000.0
    arrival_scale: float = 6.0
    class_mix: Tuple[float, float, float] = (0.25, 0.45, 0.30)
    max_queue_packets: int = 6
    power_levels_dbm: Tuple[float, ...] = (5.0, 13.0, 20.0)
    noise_w: float = 1.0e-12
    fading_rho: float = 0.88
    fading_sigma_db: float = 2.2
    shadow_sigma_db: float = 4.5
    base_gain_min_db: float = -93.0
    base_gain_max_db: float = -84.0
    burst_probability: float = 0.05
    burst_multiplier: float = 3.0
    warm_start_probability: float = 0.35
    warm_start_packets: int = 2

    # ``steps_per_episode`` is the primary measurement window H. A cohort
    # follow-up of at least max(D_k) slots is appended automatically when this
    # value is zero. Background arrivals may continue during follow-up to avoid
    # an artificial drain-only system, but they never enter primary metrics.
    deadline_followup_slots: int = 0
    continue_background_during_followup: bool = True

    # Regimes: stationary is in-distribution; robust_train uses episode-wise
    # domain randomization; mild_stress is development/validation only; stress
    # is the locked held-out final test with pre-surge, surge, degradation,
    # and recovery phases.
    regime: str = "stationary"  # stationary | robust_train | mild_stress | stress

    # Held-out stress schedule. The recovery phase returns traffic toward its
    # baseline but retains a milder residual channel impairment.
    stress_surge_start: float = 0.25
    stress_degradation_start: float = 0.55
    stress_recovery_start: float = 0.78
    stress_surge_arrival_multiplier: float = 1.55
    stress_recovery_arrival_multiplier: float = 1.00
    stress_surge_burst_probability: float = 0.13
    stress_channel_loss_db: float = -6.0
    stress_csi_error_sigma_db: float = 3.0
    stress_recovery_channel_loss_db: float = -2.0
    stress_recovery_csi_error_sigma_db: float = 1.0

    # Online recovery-monitor definitions. These are reporting controls, not
    # hidden reward parameters.
    recovery_monitor_window: int = 10
    recovery_min_emergency_arrivals: int = 3
    recovery_emergency_target: float = 0.04
    recovery_overflow_target: float = 0.03
    seed: int = 1

class ChannelAgentSpectrumEnv:
    """Sequential-reservation channel-agent spectrum-scheduling environment."""

    def __init__(self, config: ChannelEnvConfig):
        if config.regime not in {"stationary", "robust_train", "mild_stress", "stress"}:
            raise ValueError("regime must be stationary, robust_train, mild_stress, or stress.")
        self.cfg = config
        self.measurement_steps = int(config.steps_per_episode)
        if self.measurement_steps <= 0:
            raise ValueError("steps_per_episode must be positive.")
        minimum_followup = max(int(x.deadline_slots) for x in TRAFFIC_CLASSES)
        requested_followup = int(config.deadline_followup_slots)
        self.followup_steps = minimum_followup if requested_followup == 0 else requested_followup
        if self.followup_steps < minimum_followup:
            raise ValueError(
                "deadline_followup_slots must be zero (automatic) or at least "
                f"the maximum class deadline ({minimum_followup} slots)."
            )
        self.total_steps = self.measurement_steps + self.followup_steps
        self.n_devices = int(config.n_devices)
        self.n_channels = int(config.n_channels)
        self.n_classes = len(TRAFFIC_CLASSES)
        self.n_power = len(config.power_levels_dbm)
        self.action_dim = 1 + self.n_devices * self.n_power
        self.rng = np.random.default_rng(int(config.seed))
        self.t = 0
        self.done = False
        self.device_class = np.zeros(self.n_devices, dtype=int)
        self.base_gain_db = np.zeros((self.n_devices, self.n_channels), dtype=float)
        self.shadow_db = np.zeros((self.n_devices, self.n_channels), dtype=float)
        self.fast_fading_db = np.zeros((self.n_devices, self.n_channels), dtype=float)
        self.csi_error_db = np.zeros((self.n_devices, self.n_channels), dtype=float)
        self.queues: List[Deque[Packet]] = [deque() for _ in range(self.n_devices)]
        self.battery_j = np.zeros(self.n_devices, dtype=float)
        self.domain_load_mult = 1.0
        self.domain_burst_prob = float(config.burst_probability)
        self.domain_class_mult = np.ones(self.n_classes, dtype=float)
        self.domain_channel_loss_db = 0.0
        self.domain_csi_sigma_db = 0.0
        self.domain_shift_start = 0.5
        self.domain_shift_end = 0.7
        self._build_topology(int(config.seed))
        self._reset_metrics()

    # ------------------------------------------------------------------
    # topology, regimes, and reset
    # ------------------------------------------------------------------
    def _build_topology(self, seed: int) -> None:
        topo_rng = np.random.default_rng((int(seed) * 1_000_003 + 7919) % (2**32 - 1))
        mix = np.asarray(self.cfg.class_mix, dtype=float)
        mix = np.clip(mix, 1e-9, None)
        mix /= mix.sum()
        self.device_class = topo_rng.choice(self.n_classes, size=self.n_devices, p=mix)
        self.base_gain_db = topo_rng.uniform(
            self.cfg.base_gain_min_db, self.cfg.base_gain_max_db,
            size=(self.n_devices, self.n_channels),
        )
        self.shadow_db = topo_rng.normal(0.0, self.cfg.shadow_sigma_db, size=(self.n_devices, self.n_channels))
        self.fast_fading_db = np.zeros((self.n_devices, self.n_channels), dtype=float)
        self.csi_error_db = np.zeros((self.n_devices, self.n_channels), dtype=float)
        self.scenario_seed = int(seed)

    def _sample_domain_parameters(self) -> None:
        """Deterministic episode-wise mild domain randomization for training."""
        if self.cfg.regime != "robust_train":
            self.domain_load_mult = 1.0
            self.domain_burst_prob = float(self.cfg.burst_probability)
            self.domain_class_mult = np.ones(self.n_classes, dtype=float)
            self.domain_channel_loss_db = 0.0
            self.domain_csi_sigma_db = 0.0
            self.domain_shift_start, self.domain_shift_end = 0.5, 0.7
            return
        # Uses the scenario RNG after reset; hence the domain is reproducible
        # from the training episode seed and never sampled from final tests.
        self.domain_load_mult = float(self.rng.uniform(0.88, 1.10))
        self.domain_burst_prob = float(self.rng.uniform(0.03, 0.09))
        self.domain_class_mult = np.asarray([
            self.rng.uniform(0.95, 1.30),
            self.rng.uniform(0.90, 1.18),
            self.rng.uniform(0.82, 1.10),
        ], dtype=float)
        self.domain_channel_loss_db = float(self.rng.uniform(-2.0, 0.0))
        self.domain_csi_sigma_db = float(self.rng.uniform(0.0, 1.25))
        self.domain_shift_start = float(self.rng.uniform(0.38, 0.52))
        self.domain_shift_end = float(min(self.domain_shift_start + self.rng.uniform(0.12, 0.22), 0.82))

    def _regime_values(self, at_slot: Optional[int] = None) -> Tuple[float, float, float, float, np.ndarray, str]:
        """Return load, burst, channel loss, CSI sigma, class multipliers, phase."""
        slot = self.t if at_slot is None else int(at_slot)
        # Follow-up uses the terminal primary-window regime. This avoids an
        # undocumented second stress schedule while preserving background load.
        phase_slot = min(max(slot, 0), self.measurement_steps - 1)
        phase = phase_slot / max(self.measurement_steps - 1, 1)
        if self.cfg.regime == "stationary":
            return 1.0, float(self.cfg.burst_probability), 0.0, 0.0, np.ones(self.n_classes, dtype=float), "stationary"
        if self.cfg.regime == "robust_train":
            active = self.domain_shift_start <= phase < self.domain_shift_end
            load = self.domain_load_mult * (1.15 if active else 1.0)
            burst = min(0.14, self.domain_burst_prob + (0.035 if active else 0.0))
            cls = self.domain_class_mult.copy()
            loss = self.domain_channel_loss_db + (-1.0 if active else 0.0)
            csi = self.domain_csi_sigma_db + (0.55 if active else 0.0)
            if active:
                cls *= np.asarray([1.14, 1.07, 0.96], dtype=float)
            return float(load), float(burst), float(loss), float(csi), cls, "domain_shift" if active else "domain_base"
        if self.cfg.regime == "mild_stress":
            if phase < 0.32:
                return 1.0, float(self.cfg.burst_probability), 0.0, 0.0, np.ones(self.n_classes), "pre_surge"
            if phase < 0.68:
                return 1.22, 0.085, -2.5, 1.2, np.asarray([1.18, 1.08, 0.98]), "surge"
            return 1.04, float(self.cfg.burst_probability), -1.0, 0.6, np.asarray([1.06, 1.03, 1.0]), "recovery"
        # Locked strong held-out OOD stress regime.
        if phase < self.cfg.stress_surge_start:
            return 1.0, float(self.cfg.burst_probability), 0.0, 0.0, np.ones(self.n_classes), "pre_surge"
        if phase < self.cfg.stress_degradation_start:
            return (float(self.cfg.stress_surge_arrival_multiplier), float(self.cfg.stress_surge_burst_probability),
                    0.0, 0.0, np.asarray([1.35, 1.12, 0.95]), "surge")
        if phase < self.cfg.stress_recovery_start:
            return (float(self.cfg.stress_recovery_arrival_multiplier), float(self.cfg.burst_probability),
                    float(self.cfg.stress_channel_loss_db), float(self.cfg.stress_csi_error_sigma_db),
                    np.asarray([1.18, 1.08, 1.0]), "degradation")
        return (1.0, float(self.cfg.burst_probability), float(self.cfg.stress_recovery_channel_loss_db),
                float(self.cfg.stress_recovery_csi_error_sigma_db), np.asarray([1.03, 1.02, 1.0]), "recovery")

    @property
    def current_phase(self) -> str:
        return str(self._regime_values()[-1])

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is None:
            seed = int(self.cfg.seed)
        self.rng = np.random.default_rng(int(seed))
        self._build_topology(int(seed))
        self._sample_domain_parameters()
        self.t = 0
        self.done = False
        self._prepared_slot = -1
        self.queues = [deque() for _ in range(self.n_devices)]
        self.battery_j = np.asarray([TRAFFIC_CLASSES[int(c)].battery_j for c in self.device_class], dtype=float)
        self._reset_metrics()
        for i in range(self.n_devices):
            if self.rng.random() < self.cfg.warm_start_probability:
                c = int(self.device_class[i])
                profile = TRAFFIC_CLASSES[c]
                count = int(self.rng.integers(1, self.cfg.warm_start_packets + 1))
                for _ in range(count):
                    self._append_packet(i, profile, at_slot=0, metric_eligible=False, warm_start=True)
        # Primary exogenous traffic is present at the beginning of slot zero.
        # It is therefore eligible for service immediately and no artificial
        # final-slot arrival is generated after the last service opportunity.
        self._slot_arrivals, self._slot_overflow = self._generate_arrivals(at_slot=0, metric_eligible=True)
        self._pending_expired = np.zeros(self.n_classes, dtype=int)
        return self.global_state()

    def _reset_metrics(self) -> None:
        # All unprefixed packet counters below refer only to the monitored
        # primary-window arrival cohort. Warm-start and follow-up background
        # traffic have separate counters.
        self.arrivals_total = 0
        self.arrivals_by_class = np.zeros(self.n_classes, dtype=int)
        self.overflow_total = 0
        self.overflow_by_class = np.zeros(self.n_classes, dtype=int)
        self.accepted_total = 0
        self.accepted_by_class = np.zeros(self.n_classes, dtype=int)
        self.deadline_total = 0
        self.deadline_by_class = np.zeros(self.n_classes, dtype=int)
        self.completed_total = 0
        self.completed_by_class = np.zeros(self.n_classes, dtype=int)
        self.background_arrivals_total = 0
        self.background_overflow_total = 0
        self.background_deadline_total = 0
        self.background_completed_total = 0
        self.warm_start_packets_total = 0
        self.warm_start_deadline_total = 0
        self.warm_start_completed_total = 0
        self.delivered_bits = 0.0
        self.delivered_by_class = np.zeros(self.n_classes, dtype=float)
        self.energy_j = 0.0
        self.followup_delivered_bits = 0.0
        self.followup_energy_j = 0.0
        self.minimum_battery_fraction = 1.0
        self.scheduled_total = 0
        self.idle_total = 0
        self.reservation_conflicts = 0
        self.channel_use = np.zeros(self.n_channels, dtype=int)
        self.delay_ms_samples: List[float] = []
        self.action_counts = np.zeros(self.action_dim, dtype=int)
        self.power_counts = np.zeros(self.n_power, dtype=int)
        self.service_by_device = np.zeros(self.n_devices, dtype=float)
        self.last_slot_assignments = np.full(self.n_channels, -1, dtype=int)
        self.last_slot_service_bits = np.zeros(self.n_channels, dtype=float)
        self.last_slot_packet_completion = np.zeros(self.n_channels, dtype=float)
        self.last_slot_assignment_device = np.full(self.n_channels, -1, dtype=int)
        self.last_slot_assignment_power_index = np.full(self.n_channels, -1, dtype=int)
        self.last_slot_head_bits_before = np.zeros(self.n_channels, dtype=float)
        self.queue_full_device_slots = 0
        self.queue_full_any_slots = 0
        self.queue_observation_device_slots = 0
        self.queue_observation_slots = 0
        phase_names = ("pre_surge", "surge", "degradation", "recovery", "stationary", "domain_base", "domain_shift")
        self.phase_counters = {
            name: {"slots": 0, "arrivals": np.zeros(self.n_classes, dtype=int),
                   "overflow": np.zeros(self.n_classes, dtype=int), "expired": np.zeros(self.n_classes, dtype=int),
                   "completed": np.zeros(self.n_classes, dtype=int),
                   "delivered_bits": 0.0, "energy_j": 0.0,
                   "service_bits_by_class": np.zeros(self.n_classes, dtype=float),
                   "queue_backlog_packets_sum": np.zeros(self.n_classes, dtype=float),
                   "queue_backlog_samples": 0,
                   "backlog_total_trace": []}
            for name in phase_names
        }
        self.recovery_window: Deque[Tuple[int, int, int, int]] = deque(maxlen=max(int(self.cfg.recovery_monitor_window), 1))
        self.recovery_achieved = False
        self.recovery_time_slots = float("nan")
        self.recovery_slots_observed = 0
        self._slot_arrivals = np.zeros(self.n_classes, dtype=int)
        self._slot_overflow = np.zeros(self.n_classes, dtype=int)
        self._pending_expired = np.zeros(self.n_classes, dtype=int)

    # ------------------------------------------------------------------
    # observation and action interfaces
    # ------------------------------------------------------------------
    @property
    def local_obs_dim(self) -> int:
        # per device: queue, slack, battery, priority, class one-hot(3), CQI,
        # reservation = 9. Tail: channel idx, sin/cos phase, global queue, 4 duals.
        return self.n_devices * 9 + 8

    @property
    def global_state_dim(self) -> int:
        # per device: queue, slack, battery, class one-hot(3), mean CQI = 7.
        return self.n_devices * 7 + self.n_channels + 7

    def action_index(self, device: int, power_idx: int) -> int:
        return 1 + int(device) * self.n_power + int(power_idx)

    def decode_action(self, action: int) -> Tuple[int, int]:
        a = int(action)
        if a <= 0:
            return -1, -1
        a -= 1
        return int(a // self.n_power), int(a % self.n_power)

    def valid_action_mask(self, channel: int, reservation_mask: np.ndarray) -> np.ndarray:
        mask = np.zeros(self.action_dim, dtype=bool)
        mask[0] = True
        reservation_mask = np.asarray(reservation_mask, dtype=bool)
        for i in range(self.n_devices):
            if reservation_mask[i] or not self.queues[i] or self.battery_j[i] <= 0.0:
                continue
            max_power = TRAFFIC_CLASSES[int(self.device_class[i])].max_power_dbm
            for pidx, p_dbm in enumerate(self.cfg.power_levels_dbm):
                if float(p_dbm) <= max_power:
                    mask[self.action_index(i, pidx)] = True
        return mask

    def class_deadline_rates(self) -> np.ndarray:
        denom = np.maximum(self.arrivals_by_class.astype(float), 1.0)
        return self.deadline_by_class.astype(float) / denom

    def class_urgency_risk(self) -> np.ndarray:
        risk = np.zeros(self.n_classes, dtype=float)
        for queue in self.queues:
            if not queue:
                continue
            pkt = queue[0]
            c = int(pkt.class_id)
            profile = TRAFFIC_CLASSES[c]
            slack_frac = (pkt.deadline_slot - self.t) / max(profile.deadline_slots, 1)
            urgency = float(np.clip(1.0 - slack_frac, 0.0, 1.5))
            q_frac = len(queue) / max(self.cfg.max_queue_packets, 1)
            risk[c] += (profile.priority / 3.0) * (0.65 * urgency + 0.35 * q_frac)
        return risk / max(self.n_devices, 1)

    def queue_backlog_by_class(self) -> np.ndarray:
        """Current queued packets grouped by their traffic class."""
        out = np.zeros(self.n_classes, dtype=float)
        for queue in self.queues:
            for packet in queue:
                out[int(packet.class_id)] += 1.0
        return out


    def channel_observation(self, channel: int, reservation_mask: np.ndarray, dual_context: Optional[np.ndarray] = None) -> np.ndarray:
        reservation_mask = np.asarray(reservation_mask, dtype=float)
        parts: List[float] = []
        for i in range(self.n_devices):
            c = int(self.device_class[i])
            q = len(self.queues[i]) / max(self.cfg.max_queue_packets, 1)
            if self.queues[i]:
                head = self.queues[i][0]
                slack = np.clip((head.deadline_slot - self.t) / max(TRAFFIC_CLASSES[c].deadline_slots, 1), -1.0, 1.0)
            else:
                slack = 1.0
            battery = self.battery_j[i] / max(TRAFFIC_CLASSES[c].battery_j, 1e-9)
            priority = TRAFFIC_CLASSES[c].priority / 3.0
            cqi = np.tanh(self.estimated_sinr_db(i, int(channel)) / 20.0)
            onehot = [1.0 if c == k else 0.0 for k in range(self.n_classes)]
            parts.extend([q, slack, battery, priority, *onehot, cqi, float(reservation_mask[i])])
        d = np.zeros(4, dtype=float) if dual_context is None else np.asarray(dual_context, dtype=float).reshape(-1)
        if d.size < 4:
            d = np.pad(d, (0, 4 - d.size))
        d = np.clip(d[:4], 0.0, 5.0) / 5.0
        phase_clock = min(self.t, self.measurement_steps - 1)
        tail = [
            float(channel) / max(self.n_channels - 1, 1),
            np.sin(2.0 * np.pi * phase_clock / max(self.measurement_steps, 1)),
            np.cos(2.0 * np.pi * phase_clock / max(self.measurement_steps, 1)),
            float(np.mean([len(q) for q in self.queues])) / max(self.cfg.max_queue_packets, 1),
            *d.tolist(),
        ]
        return np.asarray(parts + tail, dtype=np.float32)

    def global_state(self, dual_context: Optional[np.ndarray] = None) -> np.ndarray:
        parts: List[float] = []
        for i in range(self.n_devices):
            c = int(self.device_class[i])
            q = len(self.queues[i]) / max(self.cfg.max_queue_packets, 1)
            if self.queues[i]:
                head = self.queues[i][0]
                slack = np.clip((head.deadline_slot - self.t) / max(TRAFFIC_CLASSES[c].deadline_slots, 1), -1.0, 1.0)
            else:
                slack = 1.0
            battery = self.battery_j[i] / max(TRAFFIC_CLASSES[c].battery_j, 1e-9)
            cqi = np.tanh(np.mean([self.estimated_sinr_db(i, ch) for ch in range(self.n_channels)]) / 20.0)
            onehot = [1.0 if c == k else 0.0 for k in range(self.n_classes)]
            parts.extend([q, slack, battery, *onehot, cqi])
        d = np.zeros(4, dtype=float) if dual_context is None else np.asarray(dual_context, dtype=float).reshape(-1)
        if d.size < 4:
            d = np.pad(d, (0, 4 - d.size))
        d = np.clip(d[:4], 0.0, 5.0) / 5.0
        channel_cqi = [float(np.mean(np.tanh(self.estimated_sinr_db(np.arange(self.n_devices), ch) / 20.0))) for ch in range(self.n_channels)]
        phase_clock = min(self.t, self.measurement_steps - 1)
        tail = [
            *channel_cqi,
            np.sin(2.0 * np.pi * phase_clock / max(self.measurement_steps, 1)),
            np.cos(2.0 * np.pi * phase_clock / max(self.measurement_steps, 1)),
            float(np.mean([len(q) for q in self.queues])) / max(self.cfg.max_queue_packets, 1),
            *d.tolist(),
        ]
        return np.asarray(parts + tail, dtype=np.float32)

    # ------------------------------------------------------------------
    # PHY, queue, and one-slot transition
    # ------------------------------------------------------------------
    def estimated_sinr_db(self, device: int | np.ndarray, channel: int) -> np.ndarray | float:
        idx = np.asarray(device)
        _load, _burst, channel_loss, _csi_sigma, _class_mult, _phase = self._regime_values()
        ref_power = 13.0
        signal_dbm = ref_power + self.base_gain_db[idx, channel] + self.shadow_db[idx, channel] + self.fast_fading_db[idx, channel] + channel_loss + self.csi_error_db[idx, channel]
        noise_dbm = 10.0 * np.log10(max(self.cfg.noise_w * 1000.0, 1e-15))
        out = signal_dbm - noise_dbm
        return float(out) if np.ndim(out) == 0 else out

    @staticmethod
    def _dbm_to_w(dbm: float) -> float:
        return float(10.0 ** ((float(dbm) - 30.0) / 10.0))

    def _capacity_bits(self, device: int, channel: int, power_dbm: float) -> Tuple[float, float]:
        _load, _burst, channel_loss, _csi_sigma, _class_mult, _phase = self._regime_values()
        signal_dbm = float(power_dbm + self.base_gain_db[device, channel] + self.shadow_db[device, channel] + self.fast_fading_db[device, channel] + channel_loss)
        signal_w = self._dbm_to_w(signal_dbm)
        sinr = signal_w / max(self.cfg.noise_w, 1e-15)
        se = min(np.log2(1.0 + sinr), 5.5)
        bits = float(max(0.0, se * self.cfg.channel_bandwidth_hz * self.cfg.slot_s))
        sinr_db = float(10.0 * np.log10(max(sinr, 1e-15)))
        return bits, sinr_db

    def predict_action_service(self, device: int, channel: int, power_index: int) -> Dict[str, float]:
        """Return scheduler-visible predicted service for a device-channel-power action.

        The estimate uses current CSI, including the configured CSI estimation
        error during degradation. The physical transition uses latent CSI without
        this error. This separation is intentional and the prediction error is
        logged at action level for auditability.
        """
        result = {
            "predicted_service_bits": 0.0,
            "predicted_packet_completion": 0.0,
            "predicted_head_bits": 0.0,
            "predicted_sinr_db": -100.0,
            "predicted_energy_j": 0.0,
            "predicted_queue_packets": 0.0,
        }
        if not (0 <= int(device) < self.n_devices and 0 <= int(channel) < self.n_channels and 0 <= int(power_index) < self.n_power):
            return result
        if not self.queues[int(device)] or self.battery_j[int(device)] <= 0.0:
            return result
        dev = int(device); ch = int(channel); pidx = int(power_index)
        cls = int(self.device_class[dev])
        profile = TRAFFIC_CLASSES[cls]
        power_dbm = float(min(self.cfg.power_levels_dbm[pidx], profile.max_power_dbm))
        _load, _burst, channel_loss, _csi_sigma, _class_mult, _phase = self._regime_values()
        signal_dbm = float(power_dbm + self.base_gain_db[dev, ch] + self.shadow_db[dev, ch] + self.fast_fading_db[dev, ch] + channel_loss + self.csi_error_db[dev, ch])
        signal_w = self._dbm_to_w(signal_dbm)
        sinr = signal_w / max(self.cfg.noise_w, 1e-15)
        se = min(np.log2(1.0 + sinr), 5.5)
        capacity = float(max(0.0, se * self.cfg.channel_bandwidth_hz * self.cfg.slot_s))
        head_bits = float(self.queues[dev][0].remaining_bits)
        bits = float(min(capacity, head_bits))
        circuit_w = 0.06
        base_energy = (self._dbm_to_w(power_dbm) + circuit_w) * self.cfg.slot_s
        energy = base_energy + bits * profile.energy_per_bit_j
        if energy > self.battery_j[dev]:
            available = max(self.battery_j[dev] - base_energy, 0.0)
            bits = float(min(bits, available / max(profile.energy_per_bit_j, 1e-12)))
            energy = base_energy + bits * profile.energy_per_bit_j
        result.update({
            "predicted_service_bits": float(max(bits, 0.0)),
            "predicted_packet_completion": float(np.clip(bits / max(head_bits, 1e-12), 0.0, 1.0)),
            "predicted_head_bits": head_bits,
            "predicted_sinr_db": float(10.0 * np.log10(max(sinr, 1e-15))),
            "predicted_energy_j": float(max(energy, 0.0)),
            "predicted_queue_packets": float(len(self.queues[dev])),
        })
        return result

    def _append_packet(
        self,
        device: int,
        profile: TrafficClass,
        at_slot: int,
        *,
        metric_eligible: bool,
        warm_start: bool = False,
    ) -> bool:
        """Append one packet and update the correct arrival cohort counters."""
        c = int(self.device_class[device])
        arrival_phase = str(self._regime_values(at_slot=at_slot)[-1])
        if warm_start:
            self.warm_start_packets_total += 1
        elif metric_eligible:
            self.arrivals_total += 1
            self.arrivals_by_class[c] += 1
            self.phase_counters[arrival_phase]["arrivals"][c] += 1
        else:
            self.background_arrivals_total += 1

        if len(self.queues[device]) >= self.cfg.max_queue_packets:
            if metric_eligible and not warm_start:
                self.overflow_total += 1
                self.overflow_by_class[c] += 1
                self.phase_counters[arrival_phase]["overflow"][c] += 1
            elif not warm_start:
                self.background_overflow_total += 1
            return False

        if metric_eligible and not warm_start:
            self.accepted_total += 1
            self.accepted_by_class[c] += 1
        self.queues[device].append(Packet(
            remaining_bits=float(profile.payload_bits),
            arrival_slot=int(at_slot),
            deadline_slot=int(at_slot + profile.deadline_slots),
            class_id=c,
            arrival_phase=arrival_phase,
            metric_eligible=bool(metric_eligible and not warm_start),
            warm_start=bool(warm_start),
        ))
        return True

    def _generate_arrivals(self, *, at_slot: int, metric_eligible: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Generate arrivals at the beginning of ``at_slot``.

        Returned vectors include only monitored arrivals and overflow. Follow-up
        background traffic is intentionally absent from online reliability
        denominators while remaining physically present in the queues.
        """
        arrivals = np.zeros(self.n_classes, dtype=int)
        overflow = np.zeros(self.n_classes, dtype=int)
        load_mult, burst_prob, _channel_loss, _csi_sigma, class_mult, _phase = self._regime_values(at_slot=at_slot)
        burst = self.cfg.burst_multiplier if self.rng.random() < burst_prob else 1.0
        for i in range(self.n_devices):
            c = int(self.device_class[i])
            profile = TRAFFIC_CLASSES[c]
            lam = profile.arrival_pps * self.cfg.arrival_scale * load_mult * class_mult[c] * burst * self.cfg.slot_s
            count = int(self.rng.poisson(max(lam, 0.0)))
            for _ in range(count):
                accepted = self._append_packet(i, profile, int(at_slot), metric_eligible=metric_eligible)
                if metric_eligible:
                    arrivals[c] += 1
                    overflow[c] += int(not accepted)
        return arrivals, overflow

    def prepare_slot(self) -> np.ndarray:
        """Expire overdue packets before any action is selected in this slot.

        This method is idempotent so selectors may call it defensively. Under
        the v8.9 convention a packet with ``deadline_slot == t`` is already
        overdue and is not exposed as a feasible scheduling candidate.
        """
        if self.done:
            return np.zeros(self.n_classes, dtype=int)
        if self._prepared_slot == self.t:
            return self._pending_expired.copy()
        expired = np.zeros(self.n_classes, dtype=int)
        for queue in self.queues:
            while queue and queue[0].deadline_slot <= self.t:
                packet = queue.popleft()
                c = int(packet.class_id)
                if packet.metric_eligible:
                    expired[c] += 1
                    self.deadline_total += 1
                    self.deadline_by_class[c] += 1
                    self.phase_counters[packet.arrival_phase]["expired"][c] += 1
                elif packet.warm_start:
                    self.warm_start_deadline_total += 1
                else:
                    self.background_deadline_total += 1
        self._pending_expired = expired
        self._prepared_slot = self.t
        return expired.copy()

    def _advance_fading(self) -> None:
        rho = float(np.clip(self.cfg.fading_rho, 0.0, 0.999))
        noise = self.rng.normal(0.0, self.cfg.fading_sigma_db, size=self.fast_fading_db.shape)
        self.fast_fading_db = rho * self.fast_fading_db + np.sqrt(max(1.0 - rho * rho, 0.0)) * noise
        _load, _burst, _loss, csi_sigma, _class_mult, _phase = self._regime_values()
        self.csi_error_db = self.rng.normal(0.0, csi_sigma, size=self.fast_fading_db.shape) if csi_sigma > 0.0 else np.zeros_like(self.fast_fading_db)

    def step(self, channel_actions: Sequence[int] | np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, float | np.ndarray]]:
        if self.done:
            raise RuntimeError("Episode is complete. Call reset().")
        expired = self.prepare_slot()
        actions = np.asarray(channel_actions, dtype=int).reshape(-1)
        if actions.size != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channel actions, received {actions.size}.")

        phase_name = self.current_phase
        in_measurement = bool(self.t < self.measurement_steps)
        arrivals_new = self._slot_arrivals.copy()
        overflow_new = self._slot_overflow.copy()
        reservation = np.zeros(self.n_devices, dtype=bool)
        assignments: List[Tuple[int, int, int, float]] = []
        invalid = 0
        self.last_slot_service_bits[:] = 0.0
        self.last_slot_packet_completion[:] = 0.0
        self.last_slot_assignment_device[:] = -1
        self.last_slot_assignment_power_index[:] = -1
        self.last_slot_head_bits_before[:] = 0.0
        for ch, action in enumerate(actions):
            if in_measurement:
                self.action_counts[int(np.clip(action, 0, self.action_dim - 1))] += 1
            dev, pidx = self.decode_action(int(action))
            if dev < 0:
                if in_measurement:
                    self.idle_total += 1
                continue
            if dev >= self.n_devices or pidx < 0 or pidx >= self.n_power:
                invalid += 1
                if in_measurement:
                    self.idle_total += 1
                continue
            if reservation[dev] or not self.queues[dev] or self.battery_j[dev] <= 0.0:
                invalid += 1
                if in_measurement:
                    self.idle_total += 1
                continue
            p_dbm = float(min(self.cfg.power_levels_dbm[pidx], TRAFFIC_CLASSES[int(self.device_class[dev])].max_power_dbm))
            reservation[dev] = True
            assignments.append((ch, dev, pidx, p_dbm))
            self.last_slot_assignment_device[ch] = int(dev)
            self.last_slot_assignment_power_index[ch] = int(pidx)
            self.last_slot_head_bits_before[ch] = float(self.queues[dev][0].remaining_bits)
            if in_measurement:
                self.power_counts[pidx] += 1

        if in_measurement:
            self.reservation_conflicts += int(invalid)
        delivered_slot = 0.0
        energy_slot = 0.0
        served_by_class = np.zeros(self.n_classes, dtype=float)
        for ch, dev, _pidx, p_dbm in assignments:
            if not self.queues[dev]:
                continue
            cap_bits, _sinr = self._capacity_bits(dev, ch, p_dbm)
            packet = self.queues[dev][0]
            bits = float(min(packet.remaining_bits, cap_bits))
            profile = TRAFFIC_CLASSES[int(packet.class_id)]
            circuit_w = 0.06
            energy = (self._dbm_to_w(p_dbm) + circuit_w) * self.cfg.slot_s + bits * profile.energy_per_bit_j
            if energy > self.battery_j[dev]:
                available = max(self.battery_j[dev] - (self._dbm_to_w(p_dbm) + circuit_w) * self.cfg.slot_s, 0.0)
                bits = min(bits, available / max(profile.energy_per_bit_j, 1e-12))
                energy = (self._dbm_to_w(p_dbm) + circuit_w) * self.cfg.slot_s + bits * profile.energy_per_bit_j
            if bits <= 0.0 or energy > self.battery_j[dev] + 1e-12:
                continue
            packet.remaining_bits -= bits
            self.battery_j[dev] = max(0.0, self.battery_j[dev] - energy)
            delivered_slot += bits
            energy_slot += energy
            served_by_class[packet.class_id] += bits
            if in_measurement:
                self.service_by_device[dev] += bits
            self.last_slot_service_bits[ch] = float(bits)
            if in_measurement:
                self.scheduled_total += 1
                self.channel_use[ch] += 1
            if packet.remaining_bits <= 1e-6:
                self.last_slot_packet_completion[ch] = 1.0
                completed = self.queues[dev].popleft()
                if completed.metric_eligible:
                    self.completed_total += 1
                    self.completed_by_class[int(completed.class_id)] += 1
                    self.phase_counters[completed.arrival_phase]["completed"][int(completed.class_id)] += 1
                    self.delay_ms_samples.append((self.t - completed.arrival_slot + 1) * self.cfg.slot_s * 1e3)
                elif completed.warm_start:
                    self.warm_start_completed_total += 1
                else:
                    self.background_completed_total += 1

        if in_measurement:
            self.delivered_bits += delivered_slot
            self.delivered_by_class += served_by_class
            self.energy_j += energy_slot
        else:
            self.followup_delivered_bits += delivered_slot
            self.followup_energy_j += energy_slot
        initial_battery = np.asarray([TRAFFIC_CLASSES[int(c)].battery_j for c in self.device_class], dtype=float)
        self.minimum_battery_fraction = min(
            self.minimum_battery_fraction,
            float(np.min(self.battery_j / np.maximum(initial_battery, 1e-12))),
        )
        self.last_slot_assignments = np.full(self.n_channels, -1, dtype=int)
        for ch, dev, _pidx, _power in assignments:
            self.last_slot_assignments[ch] = dev

        # Queue occupancy, throughput and energy are primary-window measures.
        if in_measurement:
            full = np.asarray([len(queue) >= self.cfg.max_queue_packets for queue in self.queues], dtype=bool)
            self.queue_full_device_slots += int(np.sum(full))
            self.queue_full_any_slots += int(np.any(full))
            self.queue_observation_device_slots += self.n_devices
            self.queue_observation_slots += 1
        pc = self.phase_counters.setdefault(phase_name, {
            "slots": 0, "arrivals": np.zeros(self.n_classes, dtype=int),
            "overflow": np.zeros(self.n_classes, dtype=int), "expired": np.zeros(self.n_classes, dtype=int),
            "completed": np.zeros(self.n_classes, dtype=int),
            "delivered_bits": 0.0, "energy_j": 0.0,
            "service_bits_by_class": np.zeros(self.n_classes, dtype=float),
            "queue_backlog_packets_sum": np.zeros(self.n_classes, dtype=float),
            "queue_backlog_samples": 0, "backlog_total_trace": [],
        })
        backlog_by_class = self.queue_backlog_by_class()
        if in_measurement:
            pc["slots"] += 1
            pc["delivered_bits"] += float(delivered_slot)
            pc["energy_j"] += float(energy_slot)
            pc["service_bits_by_class"] += served_by_class.astype(float)
            pc["queue_backlog_packets_sum"] += backlog_by_class.astype(float)
            pc["queue_backlog_samples"] += 1
            pc["backlog_total_trace"].append(float(np.sum(backlog_by_class)))

        if in_measurement and phase_name == "recovery":
            self.recovery_slots_observed += 1
            self.recovery_window.append((int(arrivals_new[0]), int(expired[0]), int(np.sum(arrivals_new)), int(np.sum(overflow_new))))
            if (not self.recovery_achieved) and len(self.recovery_window) >= max(int(self.cfg.recovery_monitor_window), 1):
                arr0 = sum(x[0] for x in self.recovery_window)
                exp0 = sum(x[1] for x in self.recovery_window)
                arr = sum(x[2] for x in self.recovery_window)
                over = sum(x[3] for x in self.recovery_window)
                emergency_rate = exp0 / max(arr0, 1)
                overflow_rate = over / max(arr, 1)
                if arr0 >= int(self.cfg.recovery_min_emergency_arrivals) and emergency_rate <= self.cfg.recovery_emergency_target and overflow_rate <= self.cfg.recovery_overflow_target:
                    self.recovery_achieved = True
                    self.recovery_time_slots = float(self.recovery_slots_observed)
        avg_q = float(np.mean([len(q) for q in self.queues]))
        reward = (
            delivered_slot / max(self.n_channels * self.cfg.channel_bandwidth_hz * self.cfg.slot_s * 4.0, 1.0)
            - 0.025 * avg_q
            - 0.10 * float(expired.sum())
            - 0.08 * float(overflow_new.sum())
            - 0.20 * energy_slot
        )
        slot_index = int(self.t)
        self.t += 1
        self.done = self.t >= self.total_steps
        if not self.done:
            self._advance_fading()
            next_is_monitored = bool(self.t < self.measurement_steps)
            if next_is_monitored or bool(self.cfg.continue_background_during_followup):
                self._slot_arrivals, self._slot_overflow = self._generate_arrivals(
                    at_slot=self.t, metric_eligible=next_is_monitored,
                )
            else:
                self._slot_arrivals = np.zeros(self.n_classes, dtype=int)
                self._slot_overflow = np.zeros(self.n_classes, dtype=int)
            self._prepared_slot = -1
            self._pending_expired = np.zeros(self.n_classes, dtype=int)
        info: Dict[str, float | np.ndarray] = {
            "delivered_bits": float(delivered_slot),
            "energy_j": float(energy_slot),
            "avg_queue_packets": avg_q,
            "scheduled_channels": float(len(assignments)),
            "idle_channels": float(self.n_channels - len(assignments)),
            "reservation_conflicts": float(invalid),
            "deadline_expired_by_class": expired.astype(float),
            "overflow_new_by_class": overflow_new.astype(float),
            "arrivals_new_by_class": arrivals_new.astype(float),
            "overflow_by_class": self.overflow_by_class.astype(float),
            "arrivals_by_class": self.arrivals_by_class.astype(float),
            "served_bits_by_class": served_by_class.astype(float),
            "queue_backlog_by_class": backlog_by_class.astype(float),
            "service_bits_by_channel": self.last_slot_service_bits.copy(),
            "packet_completion_by_channel": self.last_slot_packet_completion.copy(),
            "assignment_device_by_channel": self.last_slot_assignment_device.copy(),
            "assignment_power_index_by_channel": self.last_slot_assignment_power_index.copy(),
            "head_bits_before_by_channel": self.last_slot_head_bits_before.copy(),
            "phase_name": phase_name,
            "slot_index": float(slot_index),
            "measurement_window": float(in_measurement),
            "followup_window": float(not in_measurement),
        }
        return self.global_state(), float(reward), bool(self.done), info

    # ------------------------------------------------------------------
    # metrics
    # ------------------------------------------------------------------
    @staticmethod
    def _jain(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        if x.sum() <= 1e-12:
            return 0.0
        return float((x.sum() ** 2) / (len(x) * np.square(x).sum() + 1e-12))

    @staticmethod
    def _entropy(counts: np.ndarray) -> float:
        x = np.asarray(counts, dtype=float)
        if x.sum() <= 1e-12:
            return 0.0
        p = x[x > 0] / x.sum()
        return float(-np.sum(p * np.log(p + 1e-12)) / np.log(max(len(x), 2)))

    def metrics(self) -> Dict[str, float]:
        episode_seconds = max(self.measurement_steps * self.cfg.slot_s, 1e-12)
        offered_den = max(self.arrivals_total, 1)
        accepted_den = max(self.accepted_total, 1)
        scheduled_cap = max(self.measurement_steps * self.n_channels, 1)
        residual_by_class = np.zeros(self.n_classes, dtype=int)
        for queue in self.queues:
            for packet in queue:
                if packet.metric_eligible:
                    residual_by_class[int(packet.class_id)] += 1
        residual_total = int(residual_by_class.sum())
        conservation_error = int(
            self.arrivals_total - self.overflow_total - self.completed_total
            - self.deadline_total - residual_total
        )
        rows: Dict[str, float] = {
            "throughput_mbps": float(self.delivered_bits / episode_seconds / 1e6),
            "avg_delay_ms": float(np.mean(self.delay_ms_samples)) if self.delay_ms_samples else 0.0,
            "offered_packets": float(self.arrivals_total),
            "accepted_packets": float(self.accepted_total),
            "completed_on_time_packets": float(self.completed_total),
            "queue_overflow_packets": float(self.overflow_total),
            "deadline_expired_packets": float(self.deadline_total),
            "queue_overflow_packet_probability": float(self.overflow_total / offered_den),
            "deadline_expiration_probability": float(self.deadline_total / offered_den),
            "deadline_expiration_given_admission": float(self.deadline_total / accepted_den),
            "total_packet_failure_probability": float((self.overflow_total + self.deadline_total) / offered_den),
            "on_time_delivery_probability": float(self.completed_total / offered_den),
            # Backward-compatible alias retained for existing reporting code.
            "deadline_violation_prob": float(self.deadline_total / offered_den),
            "queue_saturation_device_slot_probability": float(self.queue_full_device_slots / max(self.queue_observation_device_slots, 1)),
            "queue_saturation_slot_probability": float(self.queue_full_any_slots / max(self.queue_observation_slots, 1)),
            "energy_j": float(self.energy_j),
            "energy_eff_bpj": float(self.delivered_bits / max(self.energy_j, 1e-12)),
            "minimum_battery_fraction": float(self.minimum_battery_fraction),
            "battery_penalty_activation_possible": float(self.minimum_battery_fraction < 0.60),
            "jain_fairness_classes": self._jain(self.delivered_by_class),
            "jain_fairness_devices": self._jain(self.service_by_device),
            "utilization": float(self.scheduled_total / scheduled_cap),
            "mean_scheduled_channels": float(self.scheduled_total / max(self.measurement_steps, 1)),
            "idle_channel_rate": float(self.idle_total / scheduled_cap),
            "reservation_conflict_rate": float(self.reservation_conflicts / scheduled_cap),
            "valid_schedule_rate": float(1.0 - self.reservation_conflicts / max(self.scheduled_total + self.reservation_conflicts, 1)),
            "channel_action_entropy": self._entropy(np.asarray([self.channel_use[ch] for ch in range(self.n_channels)], dtype=float)),
            "power_action_entropy": self._entropy(self.power_counts),
            "deadline_total": float(self.deadline_total),
            "measurement_steps": float(self.measurement_steps),
            "deadline_followup_steps": float(self.followup_steps),
            "total_simulated_steps": float(self.total_steps),
            "background_arrivals_followup": float(self.background_arrivals_total),
            "background_overflow_followup": float(self.background_overflow_total),
            "background_deadline_expired_followup": float(self.background_deadline_total),
            "warm_start_packets_excluded": float(self.warm_start_packets_total),
            "warm_start_deadline_expired_excluded": float(self.warm_start_deadline_total),
            "followup_delivered_bits": float(self.followup_delivered_bits),
            "followup_energy_j": float(self.followup_energy_j),
            "monitored_residual_packets": float(residual_total),
            "packet_outcome_conservation_error": float(conservation_error),
            "monitored_cohort_complete": float(residual_total == 0 and conservation_error == 0),
            "regime_code": 1.0 if self.cfg.regime == "stress" else 0.0,
        }
        for c in range(self.n_classes):
            denom = max(int(self.arrivals_by_class[c]), 1)
            accepted_c = max(int(self.accepted_by_class[c]), 1)
            rows[f"deadline_expiration_class{c}"] = float(self.deadline_by_class[c] / denom)
            rows[f"deadline_expiration_given_admission_class{c}"] = float(self.deadline_by_class[c] / accepted_c)
            rows[f"total_packet_failure_class{c}"] = float((self.overflow_by_class[c] + self.deadline_by_class[c]) / denom)
            rows[f"on_time_delivery_class{c}"] = float(self.completed_by_class[c] / denom)
            rows[f"deadline_violation_class{c}"] = rows[f"deadline_expiration_class{c}"]
            rows[f"queue_overflow_class{c}"] = float(self.overflow_by_class[c] / denom)
            rows[f"offered_packets_class{c}"] = float(self.arrivals_by_class[c])
            rows[f"accepted_packets_class{c}"] = float(self.accepted_by_class[c])
            rows[f"completed_on_time_packets_class{c}"] = float(self.completed_by_class[c])
            rows[f"deadline_expired_packets_class{c}"] = float(self.deadline_by_class[c])
            rows[f"monitored_residual_packets_class{c}"] = float(residual_by_class[c])
            rows[f"served_bits_class{c}"] = float(self.delivered_by_class[c])
        for phase, pc in self.phase_counters.items():
            slots = int(pc["slots"])
            if slots <= 0:
                rows[f"phase_{phase}_slots"] = 0.0
                rows[f"phase_{phase}_throughput_mbps"] = float("nan")
                rows[f"phase_{phase}_queue_overflow_packet_probability"] = float("nan")
                rows[f"phase_{phase}_deadline_violation_class0"] = float("nan")
                rows[f"phase_{phase}_deadline_expiration_probability_by_arrival_cohort"] = float("nan")
                continue
            arrivals = np.asarray(pc["arrivals"], dtype=float)
            overflow = np.asarray(pc["overflow"], dtype=float)
            expired = np.asarray(pc["expired"], dtype=float)
            completed = np.asarray(pc["completed"], dtype=float)
            rows[f"phase_{phase}_slots"] = float(slots)
            rows[f"phase_{phase}_throughput_mbps"] = float(pc["delivered_bits"] / max(slots * self.cfg.slot_s, 1e-12) / 1e6)
            rows[f"phase_{phase}_queue_overflow_packet_probability"] = float(overflow.sum() / max(arrivals.sum(), 1.0))
            rows[f"phase_{phase}_deadline_violation_class0"] = float(expired[0] / max(arrivals[0], 1.0))
            rows[f"phase_{phase}_deadline_violation_prob"] = float(expired.sum() / max(arrivals.sum(), 1.0))
            rows[f"phase_{phase}_deadline_expiration_probability_by_arrival_cohort"] = float(expired.sum() / max(arrivals.sum(), 1.0))
            rows[f"phase_{phase}_total_packet_failure_probability_by_arrival_cohort"] = float((overflow.sum() + expired.sum()) / max(arrivals.sum(), 1.0))
            rows[f"phase_{phase}_on_time_delivery_probability_by_arrival_cohort"] = float(completed.sum() / max(arrivals.sum(), 1.0))
            q_samples = max(int(pc.get("queue_backlog_samples", 0)), 1)
            backlog_mean = np.asarray(pc.get("queue_backlog_packets_sum", np.zeros(self.n_classes)), dtype=float) / q_samples
            service_bits = np.asarray(pc.get("service_bits_by_class", np.zeros(self.n_classes)), dtype=float)
            for c in range(self.n_classes):
                rows[f"phase_{phase}_deadline_expiration_class{c}_by_arrival_cohort"] = float(expired[c] / max(arrivals[c], 1.0))
                rows[f"phase_{phase}_queue_backlog_packets_class{c}"] = float(backlog_mean[c])
                rows[f"phase_{phase}_service_bits_class{c}"] = float(service_bits[c])
            trace = np.asarray(pc.get("backlog_total_trace", []), dtype=float)
            if phase == "recovery" and trace.size >= 2:
                x = np.arange(trace.size, dtype=float)
                rows["recovery_backlog_drain_slope_packets_per_slot"] = float(np.polyfit(x, trace, 1)[0])
            elif phase == "recovery":
                rows["recovery_backlog_drain_slope_packets_per_slot"] = float("nan")
        rows["recovery_achieved"] = float(self.recovery_achieved) if self.cfg.regime in {"mild_stress", "stress"} else float("nan")
        rows["recovery_time_slots"] = float(self.recovery_time_slots)
        rows["recovery_time_slots_censored"] = float(self.recovery_time_slots if self.recovery_achieved else self.recovery_slots_observed)
        rows["recovery_slots_observed"] = float(self.recovery_slots_observed)
        rows["regime_code"] = {"stationary": 0.0, "robust_train": 1.0, "mild_stress": 2.0, "stress": 3.0}.get(self.cfg.regime, -1.0)
        return rows
