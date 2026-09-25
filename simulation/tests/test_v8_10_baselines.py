from __future__ import annotations

import itertools
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import t

from channel_env import ChannelAgentSpectrumEnv, ChannelEnvConfig, TRAFFIC_CLASSES
from faithful_baselines import (METHODS, BaselineConfig, edge_utility,
                                select_baseline_actions, build_pair_table)
from experiment_v8_10 import (validate_actions, evaluate_episode, confidence, holm,
                              execute, select_v, run, source_hashes)
from protocol_v8_10 import SUITES, scenario_seed, environment_config


def small_env(seed=7, n=3, c=2):
    env = ChannelAgentSpectrumEnv(ChannelEnvConfig(n_devices=n, n_channels=c,
        steps_per_episode=12, arrival_scale=0, warm_start_probability=0, seed=seed))
    env.reset(seed=seed)
    for dev in range(n):
        cls = TRAFFIC_CLASSES[int(env.device_class[dev])]
        for _ in range(dev + 1):
            env._append_packet(dev, cls, 0, metric_eligible=True)
    return env


class ObjectiveTests(unittest.TestCase):
    def test_objective_definitions(self):
        self.assertEqual(edge_utility("MaxWeight", 3000, 500, .0002), 1.5)
        self.assertEqual(edge_utility("WeightedMaxWeight", 3000, 500, .0002, 3), 4.5)
        self.assertEqual(edge_utility("MaxRateMatching", 3000, 500, .0002), .5)
        self.assertAlmostEqual(edge_utility("LyapunovDPP-Energy", 3000, 500, .0002, dpp_v=2), 1.1)

    def test_global_objective_matches_exhaustive_joint_power_search(self):
        for seed in (7, 18, 29):
            for name in METHODS:
                if name == "EDF":
                    continue
                env = small_env(seed)
                cfg = BaselineConfig(name, 10)
                actions, _ = select_baseline_actions(env, cfg)
                def score(assignment):
                    used = set()
                    total = 0
                    for ch, action in enumerate(assignment):
                        if not action:
                            continue
                        dev, pidx = env.decode_action(action)
                        if dev in used or not env.valid_action_mask(ch, np.zeros(env.n_devices, bool))[action]:
                            return -np.inf
                        used.add(dev)
                        prediction = env.predict_action_service(dev, ch, pidx)
                        total += edge_utility(name, sum(p.remaining_bits for p in env.queues[dev]),
                            prediction["predicted_service_bits"], prediction["predicted_energy_j"],
                            TRAFFIC_CLASSES[int(env.device_class[dev])].priority, cfg.dpp_v)
                    return total
                optimum = max(score(a) for a in itertools.product(range(env.action_dim), repeat=env.n_channels))
                self.assertAlmostEqual(score(actions), optimum, places=10)

    def test_v_zero_equals_maxweight(self):
        for seed in range(10):
            env = small_env(seed)
            a, _ = select_baseline_actions(env, BaselineConfig("MaxWeight"))
            b, _ = select_baseline_actions(env, BaselineConfig("LyapunovDPP-Energy", 0))
            np.testing.assert_array_equal(a, b)

    def test_high_energy_penalty_can_idle(self):
        env = small_env()
        actions, _ = select_baseline_actions(env, BaselineConfig("LyapunovDPP-Energy", 1e12))
        self.assertTrue(np.all(actions == 0))

    def test_edf_serves_earliest_hol_deadlines(self):
        env = small_env(n=4, c=2)
        for dev, deadline in enumerate((30, 5, 10, 20)):
            env.queues[dev][0].deadline_slot = deadline
        actions, _ = select_baseline_actions(env, BaselineConfig("EDF"))
        selected = {env.decode_action(a)[0] for a in actions if a}
        self.assertEqual(selected, {1, 2})

    def test_empty_queues_idle(self):
        env = small_env()
        for q in env.queues:
            q.clear()
        for name in METHODS:
            actions, _ = select_baseline_actions(env, BaselineConfig(name))
            self.assertTrue(np.all(actions == 0))

    def test_class_power_caps_and_battery(self):
        env = small_env(n=6)
        env.battery_j[0] = 0
        for name in METHODS:
            actions, _ = select_baseline_actions(env, BaselineConfig(name))
            validate_actions(env, actions)
            for a in actions:
                if a:
                    dev, p = env.decode_action(a)
                    self.assertNotEqual(dev, 0)
                    self.assertLessEqual(env.cfg.power_levels_dbm[p], TRAFFIC_CLASSES[int(env.device_class[dev])].max_power_dbm)

    def test_predictor_matches_physics_without_csi_error(self):
        env = small_env()
        env.csi_error_db[:] = 0
        actions, details = select_baseline_actions(env, BaselineConfig("MaxWeight"))
        predicted = [r["predicted_service_bits"] for r in details]
        _, _, _, info = env.step(actions)
        np.testing.assert_allclose(predicted, info["service_bits_by_channel"], rtol=1e-12, atol=1e-9)

    def test_invalid_configuration_rejected(self):
        for v in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                BaselineConfig("LyapunovDPP-Energy", v)
        with self.assertRaises(ValueError):
            BaselineConfig("OldUrgencyHeuristic")


class ProtocolAndStatisticsTests(unittest.TestCase):
    def test_original_environment_and_sasp_are_byte_identical(self):
        root = Path(__file__).resolve().parents[1]
        expected = {
            "channel_env.py": "c02664a5925e261100840002cc28451492e25683132d7ef15d2146dbb85f047e",
            "sasp_scheduler.py": "300425744478785592ecbb13fe2627dbbbc76522cc62fa301c36ff1de867df3c",
            "protocol_v8_9.py": "54622afb9868bdd42fa22ae7e61fbfe35ba794636e85f98bb45b128a8493a6c7",
        }
        for name, checksum in expected.items():
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), checksum)

    def test_provisional_validation_cannot_unlock_final(self):
        protocol = {"source_hashes": source_hashes(), "confirmatory_eligible": False,
                    "final_plan": {"stress": {"blocks": 10, "scenarios": 72}}}
        args = SimpleNamespace(protocol="mock.json", role="final", suite="stress")
        with patch("experiment_v8_10.read_json", return_value=protocol):
            with self.assertRaisesRegex(ValueError, "Provisional"):
                run(args)

    def test_new_seeds_are_disjoint_and_outside_old_namespaces(self):
        generated = [scenario_seed(role, suite, b, i)
            for role in ("validation", "pilot", "final") for suite in SUITES
            for b in (0, 1, 99) for i in (0, 1, 999)]
        self.assertEqual(len(generated), len(set(generated)))
        self.assertGreater(min(generated), 200_000_000)
        with self.assertRaises(ValueError):
            scenario_seed("final", "stress", 100, 0)

    def test_scaling_holds_horizon_and_queue_capacity_constant(self):
        configs = [environment_config(s, 1) for s in SUITES if s.startswith("scale")]
        self.assertEqual({c.steps_per_episode for c in configs}, {180})
        self.assertEqual({c.max_queue_packets for c in configs}, {6})

    def test_ci_uses_actual_student_t_quantile(self):
        values = np.arange(12, dtype=float)
        result = confidence(values)
        half = t.ppf(.975, 11) * np.std(values, ddof=1) / np.sqrt(12)
        self.assertAlmostEqual(result["ci95_high"], np.mean(values) + half)
        self.assertIsNone(confidence([1])["ci95_low"])

    def test_holm_adjustment(self):
        np.testing.assert_allclose(holm([.01, .04, .03]), [.03, .06, .06])
        self.assertEqual(holm([None]), [1.0])

    def test_dpp_validation_feasible_and_fallback_are_explicit(self):
        def record(energy, failure):
            return {"metrics": {"deadline_expiration_class0": failure,
                "deadline_expiration_class1": 0, "deadline_expiration_class2": 0,
                "queue_overflow_packet_probability": 0, "cohort_goodput_mbps": 1,
                "energy_j": energy, "total_packet_failure_probability": failure}}
        v, mode, _ = select_v({0: [record(2, .01)], 1: [record(1, .01)]})
        self.assertEqual(v, 1)
        self.assertIn("minimum energy", mode)
        v, mode, _ = select_v({0: [record(2, .10)], 1: [record(1, .20)]})
        self.assertEqual(v, 0)
        self.assertIn("NOT target-feasible", mode)

    def test_duplicate_device_rejected(self):
        env = small_env()
        action = env.action_index(0, 0)
        with self.assertRaises(AssertionError):
            validate_actions(env, [action, action])


class IntegrationTests(unittest.TestCase):
    def test_all_methods_share_exogenous_randomness_and_complete_cohort(self):
        config = ChannelEnvConfig(n_devices=8, n_channels=3, steps_per_episode=18,
                                   regime="stress", seed=310000123)
        with patch("experiment_v8_10.environment_config", return_value=config):
            rows = [evaluate_episode("stress", 310000123, m, 1)
                    for m in ("SASP",) + METHODS]
            again = evaluate_episode("stress", 310000123, "MaxWeight", 1)
        self.assertEqual(len({r["exogenous_sha256"] for r in rows}), 1)
        mw = next(r for r in rows if r["method"] == "MaxWeight")
        self.assertEqual(again["actions_sha256"], mw["actions_sha256"])
        for r in rows:
            metrics = r["metrics"]
            self.assertEqual(metrics["packet_outcome_conservation_error"], 0)
            self.assertEqual(metrics["monitored_residual_packets"], 0)
            self.assertEqual(metrics["reservation_conflict_rate"], 0)
            self.assertEqual(metrics["monitored_cohort_complete"], 1)
            expected = sum(metrics[f"completed_on_time_packets_class{c}"] * p.payload_bits
                           for c, p in enumerate(TRAFFIC_CLASSES)) / .018 / 1e6
            self.assertAlmostEqual(metrics["cohort_goodput_mbps"], expected)

    def test_resume_does_not_rerun_and_rejects_changed_specification(self):
        cfg = ChannelEnvConfig(n_devices=4, n_channels=2, steps_per_episode=12,
                                regime="stress", seed=310000000)
        with tempfile.TemporaryDirectory() as temp, patch("experiment_v8_10.environment_config", return_value=cfg):
            execute(temp, "pilot", "stress", 1, 1, ("SASP", "EDF"), 1, "test")
            with patch("experiment_v8_10.evaluate_episode", side_effect=AssertionError("Should resume")):
                execute(temp, "pilot", "stress", 1, 1, ("SASP", "EDF"), 1, "test")
            with self.assertRaises(ValueError):
                execute(temp, "pilot", "stress", 1, 1, ("SASP", "EDF"), 2, "test")


if __name__ == "__main__":
    unittest.main()
