"""Frozen method names, configurations, and scenario namespaces for v8.9."""
from __future__ import annotations

from typing import List

from sasp_scheduler import SASPConfig


PROPOSED = "SASP"
SASP_ABLATIONS = (
    "SASP-NoPredictedService",
    "SASP-NoVirtualQueue",
    "SASP-NoCriticalCoverage",
    "SASP-GreedyAssignment",
)

SEED_OFFSETS = {
    "development": 40_000_000,
    "validation": 60_000_000,
    "pilot_test": 80_000_000,
    "confirmatory_test": 100_000_000,
    "scalability_test": 120_000_000,
}


def split_scenarios(exp_seed: int, n: int, role: str) -> List[int]:
    if role not in SEED_OFFSETS:
        raise KeyError(role)
    base = SEED_OFFSETS[role] + int(exp_seed) * 10_000
    return [base + index for index in range(int(n))]


def test_role_for(mode: str) -> str:
    if mode in {"quick", "diagnostic"}:
        return "development"
    if mode == "pilot":
        return "pilot_test"
    if mode == "final":
        return "confirmatory_test"
    raise KeyError(mode)


def is_sasp(name: str) -> bool:
    return name == PROPOSED or name in SASP_ABLATIONS


def sasp_config_for(name: str) -> SASPConfig:
    """Return full SASP or a variant that changes exactly one switch."""
    if not is_sasp(name):
        raise KeyError(name)
    values: dict[str, object] = {}
    if name == "SASP-NoPredictedService":
        values["use_predicted_service"] = False
    elif name == "SASP-NoVirtualQueue":
        values["virtual_queue_enabled"] = False
    elif name == "SASP-NoCriticalCoverage":
        values["critical_coverage_enabled"] = False
    elif name == "SASP-GreedyAssignment":
        values["assignment_solver"] = "greedy"
    return SASPConfig(**values)
