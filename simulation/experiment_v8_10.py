"""Audited, resumable nonlearning comparison. See README.md before final runs."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy
from scipy.stats import t as student_t

from channel_env import ChannelAgentSpectrumEnv, TRAFFIC_CLASSES
from faithful_baselines import BaselineConfig, select_baseline_actions
from protocol_v8_9 import is_sasp, sasp_config_for
from protocol_v8_10 import (VERSION, SUITES, V_GRID, TARGETS, PRIMARY_METRICS,
    REPORT_METRICS, scenario_seed, environment_config, configuration_record, methods_for)
from sasp_scheduler import select_sasp_actions, _virtual_queue_step

ROOT = Path(__file__).resolve().parent


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def encoded(value):
    return json.dumps(clean(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def source_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.glob("*.py"))}


def runtime():
    return {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(), "processor": platform.processor(),
            "executable": sys.executable}


def validate_actions(env, actions):
    if len(actions) != env.n_channels:
        raise AssertionError("Wrong number of channel actions")
    used = np.zeros(env.n_devices, dtype=bool)
    for ch, a in enumerate(actions):
        if not 0 <= int(a) < env.action_dim or not env.valid_action_mask(ch, used)[int(a)]:
            raise AssertionError(f"Infeasible action at slot {env.t}, channel {ch}: {a}")
        if int(a):
            dev, _ = env.decode_action(int(a))
            used[dev] = True


def evaluate_episode(suite, seed, method, dpp_v, collect_trace=False):
    env = ChannelAgentSpectrumEnv(environment_config(suite, seed))
    env.reset(seed=seed)
    saspcfg = sasp_config_for(method) if is_sasp(method) else None
    basecfg = None if saspcfg else BaselineConfig(method, dpp_v)
    vq = np.zeros(4)
    controller_ns, decision_ns, trace = [], [], []
    exogenous, action_digest = hashlib.sha256(), hashlib.sha256()
    exogenous.update(env.device_class.tobytes())
    exogenous.update(env.base_gain_db.tobytes())
    exogenous.update(env.shadow_db.tobytes())
    predicted_sum, error_sum, abs_error_sum, choices = 0.0, 0.0, 0.0, 0
    started = time.perf_counter()
    while not env.done:
        env.prepare_slot()
        slot = env.t
        exogenous.update(encoded(env.rng.bit_generator.state))
        exogenous.update(env.fast_fading_db.tobytes())
        exogenous.update(env.csi_error_db.tobytes())
        exogenous.update(env._slot_arrivals.tobytes())
        vq_before = vq.copy()
        start = time.perf_counter_ns()
        if saspcfg:
            actions, telemetry = select_sasp_actions(env, vq, saspcfg)
        else:
            actions, telemetry = select_baseline_actions(env, basecfg)
        elapsed = time.perf_counter_ns() - start
        validate_actions(env, actions)
        action_digest.update(np.asarray(actions, dtype="<i8").tobytes())
        predicted = []
        for ch, action in enumerate(actions):
            dev, pidx = env.decode_action(int(action))
            prediction = env.predict_action_service(dev, ch, pidx) if action else {}
            predicted.append(float(prediction.get("predicted_service_bits", 0.0)))
        snapshot = None
        if collect_trace:
            snapshot = {"slot": slot, "phase": env.current_phase,
                "measurement": slot < env.measurement_steps,
                "queue_bits": [sum(p.remaining_bits for p in q) for q in env.queues],
                "hol_deadlines": [q[0].deadline_slot if q else None for q in env.queues],
                "actions": list(actions), "predicted_service_bits": predicted,
                "virtual_queue_before": vq_before, "selection": telemetry}
        _, _, _, info = env.step(actions)
        update_time = 0
        if saspcfg and saspcfg.virtual_queue_enabled:
            start = time.perf_counter_ns()
            vq = _virtual_queue_step(vq, info["arrivals_new_by_class"],
                 info["deadline_expired_by_class"], info["overflow_new_by_class"], saspcfg)
            update_time = time.perf_counter_ns() - start
        # Exclude the first five primary slots from timing summaries only.
        if 5 <= slot < env.measurement_steps:
            decision_ns.append(elapsed)
            controller_ns.append(elapsed + update_time)
        realized = np.asarray(info["service_bits_by_channel"])
        if slot < env.measurement_steps:
            error = realized - np.asarray(predicted)
            chosen = np.asarray(actions) > 0
            predicted_sum += sum(predicted)
            error_sum += float(np.sum(error[chosen]))
            abs_error_sum += float(np.sum(np.abs(error[chosen])))
            choices += int(np.sum(chosen))
        if snapshot is not None:
            snapshot.update(realized_service_bits=realized, virtual_queue_after=vq.copy(),
                arrivals=info["arrivals_new_by_class"], expired=info["deadline_expired_by_class"],
                overflow=info["overflow_new_by_class"])
            trace.append(snapshot)
    metrics = env.metrics()
    for field in ("packet_outcome_conservation_error", "monitored_residual_packets", "reservation_conflict_rate"):
        if metrics[field] != 0:
            raise AssertionError(f"Failed episode audit: {field}={metrics[field]}")
    if metrics["monitored_cohort_complete"] != 1:
        raise AssertionError("Cohort incomplete")
    bits = sum(metrics[f"completed_on_time_packets_class{c}"] * cls.payload_bits
               for c, cls in enumerate(TRAFFIC_CLASSES))
    metrics["cohort_goodput_mbps"] = bits / (env.measurement_steps * env.cfg.slot_s * 1e6)
    metrics["cohort_on_time_payload_bits"] = bits
    # Do not call a class with no offered packets a zero-failure success.
    for c in range(env.n_classes):
        if metrics[f"offered_packets_class{c}"] == 0:
            for prefix in ("deadline_expiration_class", "total_packet_failure_class",
                           "on_time_delivery_class", "queue_overflow_class", "deadline_violation_class"):
                metrics[f"{prefix}{c}"] = None
    for label, values in (("controller", controller_ns), ("decision", decision_ns)):
        metrics[f"{label}_median_ms"] = float(np.median(values) / 1e6)
        metrics[f"{label}_p95_ms"] = float(np.percentile(values, 95) / 1e6)
    metrics.update(episode_wall_seconds=time.perf_counter() - started,
        selected_service_prediction_bias_bits=error_sum / max(choices, 1),
        selected_service_prediction_mae_bits=abs_error_sum / max(choices, 1),
        predicted_service_bits_primary=predicted_sum)
    return clean({"suite": suite, "scenario_seed": seed, "method": method, "dpp_v": dpp_v,
        "metrics": metrics, "exogenous_sha256": exogenous.hexdigest(),
        "actions_sha256": action_digest.hexdigest(), "trace": trace})


def finite_mean(values):
    values = [float(x) for x in values if x is not None and np.isfinite(x)]
    return float(np.mean(values)) if values else None


def confidence(values):
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    n = len(values)
    if not n:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "n_blocks": 0}
    mean = float(np.mean(values))
    half = float(student_t.ppf(.975, n - 1) * np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else None
    return {"mean": mean, "ci95_low": mean - half if half is not None else None,
            "ci95_high": mean + half if half is not None else None, "n_blocks": n}


def holm(pvalues):
    """Holm adjustment; missing p-values are conservatively treated as one."""
    p = np.array([1.0 if x is None else x for x in pvalues], dtype=float)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted.tolist()


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(clean(rows))


def summarize(records, directory, methods):
    rows = [{"method": r["method"], "block": r["block"], "index": r["index"],
             "scenario_seed": r["scenario_seed"], **r["metrics"]} for r in records]
    write_csv(Path(directory) / "scenario_metrics.csv", rows)
    blocks = sorted({r["block"] for r in records})
    summaries, pairs = [], []
    for method in methods:
        for metric in REPORT_METRICS:
            blockmeans = [finite_mean(r["metrics"].get(metric) for r in records
                          if r["method"] == method and r["block"] == b) for b in blocks]
            summaries.append({"method": method, "metric": metric, **confidence(blockmeans)})
    lookup = {(r["method"], r["block"], r["index"]): r for r in records}
    for method in methods:
        if method == "SASP":
            continue
        for metric in PRIMARY_METRICS:
            differences = []
            matched_scenarios = 0
            for b in blocks:
                delta = []
                for key, sasp in lookup.items():
                    if key[0] != "SASP" or key[1] != b:
                        continue
                    other = lookup.get((method, b, key[2]))
                    if other is None:
                        raise AssertionError("Unpaired comparison")
                    x, y = sasp["metrics"].get(metric), other["metrics"].get(metric)
                    if x is not None and y is not None:
                        delta.append(x - y)
                matched_scenarios += len(delta)
                differences.append(finite_mean(delta))
            stats = confidence(differences)
            available = [x for x in differences if x is not None]
            # Degenerate all-identical block differences do not establish a
            # sampling variance: do not manufacture a significant p-value.
            p = None
            if len(available) > 1 and np.std(available, ddof=1) > 0:
                statistic = np.mean(available) / (np.std(available, ddof=1) / np.sqrt(len(available)))
                p = float(2 * student_t.sf(abs(statistic), len(available) - 1))
            pairs.append({"comparison": f"SASP minus {method}", "metric": metric,
                **stats, "matched_scenarios": matched_scenarios, "p_two_sided": p})
    for pair, adjusted in zip(pairs, holm([r["p_two_sided"] for r in pairs])):
        pair["p_holm_within_suite"] = adjusted
    write_csv(Path(directory) / "summary.csv", summaries)
    write_csv(Path(directory) / "paired_comparisons.csv", pairs)
    return summaries


def execute(directory, role, suite, blocks, scenarios, methods, dpp_v, protocol_hash, label=""):
    directory = Path(directory)
    manifest = {"version": VERSION, "role": role, "suite": suite, "blocks": blocks,
        "scenarios_per_block": scenarios, "methods": list(methods), "dpp_v": dpp_v,
        "source_hashes": source_hashes(), "protocol_sha256": protocol_hash,
        "runtime": runtime(), "label": label}
    manifest_path = directory / "run_manifest.json"
    if manifest_path.exists():
        old = read_json(manifest_path)
        if old["specification"] != manifest:
            raise ValueError("Resume refused: run configuration, source, runtime or protocol changed. Use a new directory.")
    else:
        save_json(manifest_path, {"started_utc": utc_now(), "specification": manifest})
    run_hash = digest(manifest)
    records = []
    for b in range(blocks):
        for i in range(scenarios):
            seed = scenario_seed(role, suite, b, i)
            expected_exogenous = None
            # Rotate method execution order to reduce systematic timing order bias.
            offset = (b * scenarios + i) % len(methods)
            order = list(methods[offset:]) + list(methods[:offset])
            for method in order:
                path = directory / "episodes" / f"b{b:03d}_s{i:04d}_{method}.json"
                if path.exists():
                    envelope = read_json(path)
                    row = envelope["record"]
                    if envelope["record_sha256"] != digest(row) or row["run_sha256"] != run_hash:
                        raise ValueError(f"Episode checksum/run mismatch: {path}")
                else:
                    row = evaluate_episode(suite, seed, method, dpp_v, collect_trace=i == 0)
                    row.update(block=b, index=i, run_sha256=run_hash)
                    save_json(path, {"record_sha256": digest(row), "record": row})
                if expected_exogenous is None:
                    expected_exogenous = row["exogenous_sha256"]
                elif row["exogenous_sha256"] != expected_exogenous:
                    raise AssertionError("Common-random-number audit failed")
                if row["scenario_seed"] != seed or row["method"] != method:
                    raise AssertionError("Episode identity mismatch")
                records.append(row)
            print(f"{label or role} {suite}: block {b+1}/{blocks}, scenario {i+1}/{scenarios} complete", flush=True)
    summaries = summarize(records, directory, methods)
    save_json(directory / "completion.json", {"completed_utc": utc_now(), "episodes": len(records),
        "all_crn_checks_passed": True, "all_packet_conservation_checks_passed": True,
        "all_action_feasibility_checks_passed": True, "run_sha256": run_hash,
        "interpretation": "Pilot/development only" if role != "final" else "Frozen final nonlearning comparison"})
    return records, summaries


def select_v(records_by_v):
    summaries = []
    metrics = ("deadline_expiration_class0", "deadline_expiration_class1",
               "deadline_expiration_class2", "queue_overflow_packet_probability")
    for v, records in records_by_v.items():
        means = {m: finite_mean(r["metrics"].get(m) for r in records)
                 for m in metrics + ("cohort_goodput_mbps", "energy_j", "total_packet_failure_probability")}
        if any(means[m] is None for m in means):
            raise ValueError("Validation lacks observations required by selection rule")
        summaries.append({"v": v, **means})
    reference = next(x["cohort_goodput_mbps"] for x in summaries if x["v"] == 0)
    for row in summaries:
        violations = [max(row[m] / target - 1, 0) for m, target in zip(metrics, TARGETS)]
        violations.append(max(.95 - row["cohort_goodput_mbps"] / max(reference, 1e-12), 0) / .95)
        row["maximum_normalized_excess"] = max(violations)
        row["validation_feasible"] = max(violations) <= 1e-12
    feasible = [r for r in summaries if r["validation_feasible"]]
    if feasible:
        best = min(feasible, key=lambda r: (r["energy_j"], -r["cohort_goodput_mbps"], r["v"]))
        mode = "minimum energy among validation-feasible candidates"
    else:
        best = min(summaries, key=lambda r: (r["maximum_normalized_excess"],
                   r["total_packet_failure_probability"], r["energy_j"], r["v"]))
        mode = "fallback: least maximum normalized violation; NOT target-feasible"
    return best["v"], mode, summaries


def validate(args):
    root = Path(args.output)
    validation_spec = {"source_hashes": source_hashes(), "grid": V_GRID,
                       "selection_rule": "mean_targets_and_95pct_V0_cohort_goodput_v1"}
    rows_by_v = {}
    for v in V_GRID:
        rows_by_v[v] = []
        for suite in ("validation_stationary", "validation_mild"):
            records, _ = execute(root / f"V_{v:g}" / suite, "validation", suite,
                args.blocks, args.scenarios, ("LyapunovDPP-Energy",), v,
                digest(validation_spec), label=f"validate V={v:g}")
            rows_by_v[v].extend(records)
    chosen, mode, summaries = select_v(rows_by_v)
    reference_exogenous = {(r["suite"], r["block"], r["index"]): r["exogenous_sha256"]
                           for r in rows_by_v[0.0]}
    for records in rows_by_v.values():
        for row in records:
            if row["exogenous_sha256"] != reference_exogenous[row["suite"], row["block"], row["index"]]:
                raise AssertionError("Validation candidates did not receive identical exogenous randomness")
    result = {"created_utc": utc_now(), "version": VERSION, "selected_v": chosen,
        "selection_mode": mode, "candidate_summaries": summaries,
        "blocks": args.blocks, "scenarios_per_suite_per_block": args.scenarios,
        "confirmatory_eligible": args.blocks >= 3 and args.scenarios >= 8,
        "source_hashes": source_hashes(), "validation_specification": validation_spec,
        "record_set_sha256": digest({str(v): [digest(r) for r in rows] for v, rows in rows_by_v.items()})}
    save_json(root / "validation_selection.json", result)
    print(json.dumps({"selected_v": chosen, "selection_mode": mode,
                      "confirmatory_eligible": result["confirmatory_eligible"]}, indent=2))


def freeze(args):
    destination = Path(args.output)
    if destination.exists():
        raise ValueError("Protocol already exists; refusing to overwrite a freeze")
    if args.blocks < 5 or args.scenarios < 24 or args.support_blocks < 3 or args.support_scenarios < 8:
        raise ValueError("Frozen plan minimum: main 5 blocks x 24 scenarios; support 3 blocks x 8 scenarios")
    validation = read_json(args.validation)
    if validation["source_hashes"] != source_hashes():
        raise ValueError("Code changed since validation; revalidate")
    protocol = {"version": VERSION, "frozen_utc": utc_now(), "source_hashes": source_hashes(),
        "validation_sha256": digest(validation), "selected_v": validation["selected_v"],
        "selection_mode": validation["selection_mode"],
        "confirmatory_eligible": validation["confirmatory_eligible"],
        "environments": configuration_record(), "primary_metrics": PRIMARY_METRICS,
        "final_plan": {s: {"blocks": args.blocks if s in ("stress", "stationary") else args.support_blocks,
            "scenarios": args.scenarios if s in ("stress", "stationary") else args.support_scenarios,
            "methods": methods_for(s)} for s in SUITES if not s.startswith("validation")},
        "statistical_rule": "paired block t intervals; Holm across 5 comparators x 4 primary endpoints per suite; support suites exploratory",
        "scope": "nonlearning baseline repair; no retrained or recertified learned comparators"}
    save_json(destination, protocol)
    print(f"Created protocol {destination}; final-run eligible: {protocol['confirmatory_eligible']}")


def run(args):
    protocol = read_json(args.protocol)
    if protocol["source_hashes"] != source_hashes():
        raise ValueError("Code differs from frozen protocol. Revalidate and create a new protocol.")
    if args.suite not in protocol["final_plan"]:
        raise ValueError("Suite not frozen")
    plan = protocol["final_plan"][args.suite]
    if args.role == "final":
        if not protocol["confirmatory_eligible"]:
            raise ValueError("Provisional validation cannot unlock a final run. Use >=3 blocks and >=8 scenarios per suite.")
        if args.blocks is not None or args.scenarios is not None:
            raise ValueError("Final sample sizes come only from the frozen protocol")
        blocks, scenarios = plan["blocks"], plan["scenarios"]
    else:
        blocks, scenarios = args.blocks or 2, args.scenarios or 3
    _, summary = execute(args.output, args.role, args.suite, blocks, scenarios,
        tuple(plan["methods"]), protocol["selected_v"], digest(protocol))
    for row in summary:
        if row["metric"] in PRIMARY_METRICS:
            print(f"{row['method']:22} {row['metric']:40} {row['mean']}")


def bounded_count(value):
    number = int(value)
    if not 1 <= number <= 99:
        raise argparse.ArgumentTypeError("Count must be between 1 and 99")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("validate")
    p.add_argument("--output", required=True)
    p.add_argument("--blocks", type=bounded_count, default=3)
    p.add_argument("--scenarios", type=bounded_count, default=8)
    p.set_defaults(func=validate)
    p = sub.add_parser("freeze")
    p.add_argument("--validation", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--blocks", type=bounded_count, default=10)
    p.add_argument("--scenarios", type=bounded_count, default=72)
    p.add_argument("--support-blocks", type=bounded_count, default=5)
    p.add_argument("--support-scenarios", type=bounded_count, default=24)
    p.set_defaults(func=freeze)
    p = sub.add_parser("run")
    p.add_argument("--protocol", required=True)
    p.add_argument("--role", choices=("pilot", "final"), required=True)
    p.add_argument("--suite", choices=[s for s in SUITES if not s.startswith("validation")], default="stress")
    p.add_argument("--output", required=True)
    p.add_argument("--blocks", type=bounded_count)
    p.add_argument("--scenarios", type=bounded_count)
    p.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
