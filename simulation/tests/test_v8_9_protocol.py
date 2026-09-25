from __future__ import annotations

import itertools
import unittest
from collections import defaultdict

import numpy as np

from channel_env import ChannelAgentSpectrumEnv, ChannelEnvConfig, TRAFFIC_CLASSES
from protocol_v8_9 import SASP_ABLATIONS, PROPOSED, sasp_config_for, split_scenarios
from sasp_scheduler import _hungarian_max, evaluate_sasp


class DeadlineAndCohortTests(unittest.TestCase):
    def make_empty_env(self) -> ChannelAgentSpectrumEnv:
        env = ChannelAgentSpectrumEnv(ChannelEnvConfig(
            n_devices=6,
            n_channels=2,
            steps_per_episode=20,
            arrival_scale=0.0,
            warm_start_probability=0.0,
            regime="stationary",
            seed=7,
        ))
        env.reset(seed=7)
        return env

    def test_exclusive_deadline_expires_before_slot_d(self) -> None:
        env = self.make_empty_env()
        dev = int(np.flatnonzero(env.device_class == 0)[0])
        self.assertTrue(env._append_packet(dev, TRAFFIC_CLASSES[0], 0, metric_eligible=True))
        env.t = TRAFFIC_CLASSES[0].deadline_slots - 1
        self.assertEqual(int(env.prepare_slot().sum()), 0)
        env._prepared_slot = -1
        env.t = TRAFFIC_CLASSES[0].deadline_slots
        expired = env.prepare_slot()
        self.assertEqual(int(expired[0]), 1)
        self.assertEqual(len(env.queues[dev]), 0)

    def test_warm_start_is_excluded_from_offered_denominator(self) -> None:
        env = self.make_empty_env()
        before = env.arrivals_total
        self.assertTrue(env._append_packet(
            0, TRAFFIC_CLASSES[0], 0, metric_eligible=False, warm_start=True,
        ))
        self.assertEqual(env.arrivals_total, before)
        self.assertEqual(env.warm_start_packets_total, 1)


class AssignmentTests(unittest.TestCase):
    def test_hungarian_matches_bruteforce(self) -> None:
        rng = np.random.default_rng(19)
        for rows in range(1, 5):
            for cols in range(rows, rows + 3):
                for _ in range(20):
                    matrix = rng.normal(size=(rows, cols))
                    got = _hungarian_max(matrix)
                    got_value = sum(matrix[r, got[r]] for r in range(rows))
                    optimum = max(
                        sum(matrix[r, perm[r]] for r in range(rows))
                        for perm in itertools.permutations(range(cols), rows)
                    )
                    self.assertAlmostEqual(float(got_value), float(optimum), places=10)

class ProtocolTests(unittest.TestCase):
    def test_seed_roles_are_disjoint(self) -> None:
        roles = ["development", "validation", "pilot_test", "confirmatory_test", "scalability_test"]
        all_sets = {
            role: {
                seed
                for replicate in (1, 11, 505, 901)
                for seed in split_scenarios(replicate, 72, role)
            }
            for role in roles
        }
        for index, left in enumerate(roles):
            for right in roles[index + 1:]:
                self.assertFalse(all_sets[left] & all_sets[right], (left, right))
        self.assertFalse(any(
            10_000_000 <= seed < 40_000_000
            for values in all_sets.values() for seed in values
        ))

    def test_sasp_ablations_change_one_switch(self) -> None:
        full = sasp_config_for(PROPOSED).__dict__
        expected = {
            "SASP-NoPredictedService": {"use_predicted_service"},
            "SASP-NoVirtualQueue": {"virtual_queue_enabled"},
            "SASP-NoCriticalCoverage": {"critical_coverage_enabled"},
            "SASP-GreedyAssignment": {"assignment_solver"},
        }
        self.assertEqual(set(SASP_ABLATIONS), set(expected))
        for name, changed_expected in expected.items():
            variant = sasp_config_for(name).__dict__
            changed = set()
            for key in full:
                left, right = full[key], variant[key]
                differs = not np.array_equal(left, right) if isinstance(left, np.ndarray) else left != right
                if differs:
                    changed.add(key)
            self.assertEqual(changed, changed_expected)

    def test_small_sasp_run_is_clean_and_feasible(self) -> None:
        profile = {
            "n_devices": 8,
            "n_channels": 3,
            "steps": 18,
            "arrival_scale": 4.5,
            "max_queue_packets": 5,
        }
        factory = lambda seed: ChannelAgentSpectrumEnv(ChannelEnvConfig(
            n_devices=int(profile["n_devices"]),
            n_channels=int(profile["n_channels"]),
            steps_per_episode=int(profile["steps"]),
            arrival_scale=float(profile["arrival_scale"]),
            max_queue_packets=int(profile["max_queue_packets"]),
            regime="stationary",
            seed=int(seed),
        ))
        rows, traces = evaluate_sasp(
            factory, [40_010_001], sasp_config_for(PROPOSED),
            collect_action_trace=True,
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["monitored_cohort_complete"], 1.0)
        self.assertEqual(row["monitored_residual_packets"], 0.0)
        self.assertEqual(row["packet_outcome_conservation_error"], 0.0)
        self.assertAlmostEqual(
            row["on_time_delivery_probability"] + row["total_packet_failure_probability"],
            1.0,
        )
        by_slot: dict[tuple[int, int], list[int]] = defaultdict(list)
        for trace in traces:
            self.assertAlmostEqual(trace["policy_component"], 0.0)
            self.assertAlmostEqual(trace["checkpoint_dual_score_contribution"], 0.0)
            self.assertEqual(trace["assignment_solver_hungarian"], 1.0)
            dev = int(trace["final_device"])
            if dev >= 0:
                by_slot[(int(trace["episode_seed"]), int(trace["slot"]))].append(dev)
        for devices in by_slot.values():
            self.assertEqual(len(devices), len(set(devices)))


if __name__ == "__main__":
    unittest.main()
