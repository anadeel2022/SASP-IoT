"""Versioned, disjoint evaluation roles. No reuse of v8.9 reported scenarios."""
from dataclasses import asdict
from channel_env import ChannelEnvConfig
from faithful_baselines import METHODS
from protocol_v8_9 import SASP_ABLATIONS

VERSION = "8.10-baseline-repair"
METHOD_ORDER = ("SASP",) + METHODS
V_GRID = (0.0, 0.1, 1.0, 10.0, 100.0)
TARGETS = (0.04, 0.05, 0.20, 0.03)
PRIMARY_METRICS = ("deadline_expiration_class0", "queue_overflow_packet_probability",
                   "cohort_goodput_mbps", "energy_j")
REPORT_METRICS = PRIMARY_METRICS + ("throughput_mbps", "total_packet_failure_probability",
    "deadline_expiration_class1", "deadline_expiration_class2", "controller_median_ms",
    "controller_p95_ms", "decision_median_ms", "decision_p95_ms")
SUITES = {
    "stress": (24, 5, 180, "stress"),
    "stationary": (24, 5, 180, "stationary"),
    "long_stress": (24, 5, 900, "stress"),
    "scale12": (12, 3, 180, "stress"),
    "scale16": (16, 4, 180, "stress"),
    "scale24": (24, 5, 180, "stress"),
    "scale32": (32, 6, 180, "stress"),
    "ablation": (24, 5, 180, "stress"),
    "validation_stationary": (24, 5, 180, "stationary"),
    "validation_mild": (24, 5, 180, "mild_stress"),
}
ROLE_BASE = {"validation": 210_000_000, "pilot": 310_000_000, "final": 410_000_000}


def scenario_seed(role, suite, block, index):
    if role not in ROLE_BASE or suite not in SUITES:
        raise ValueError("Unknown seed role or suite")
    if not 0 <= block < 100 or not 0 <= index < 1000:
        raise ValueError("Seed allocation requires block <100 and scenario index <1000")
    return ROLE_BASE[role] + list(SUITES).index(suite) * 1_000_000 + block * 1000 + index


def environment_config(suite, seed):
    n, c, h, regime = SUITES[suite]
    return ChannelEnvConfig(n_devices=n, n_channels=c, steps_per_episode=h,
                            max_queue_packets=6, regime=regime, seed=int(seed))


def methods_for(suite):
    return ("SASP",) + SASP_ABLATIONS if suite == "ablation" else METHOD_ORDER


def configuration_record():
    return {suite: asdict(environment_config(suite, 0)) for suite in SUITES}
