#!/usr/bin/env python3
"""Generate Figures 2--5 from complete, frozen v8.10 evaluation records.

Inputs may be directories or ZIPs (including Windows backslash ZIP names).
No scheduler is run, no simulator modules are imported, and inputs are read-only.
Python >=3.10; dependencies: numpy, scipy, matplotlib. See README.md.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import platform
import shutil
import sys
import zipfile

import numpy as np
import scipy
from scipy.stats import t as student_t
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

VERSION = "8.10-plots-2"
METHODS = ["MaxRateMatching", "MaxWeight", "WeightedMaxWeight", "EDF",
           "LyapunovDPP-Energy", "SASP"]
LABELS = dict(zip(METHODS, ["MaxRate", "MaxWeight", "W-MaxWeight", "EDF", "DPP-Energy", "SASP"]))
COLORS = dict(zip(METHODS, ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#161616"]))
MARKERS = dict(zip(METHODS, ["o", "s", "^", "D", "v", "P"]))
ABLATIONS = ["SASP-NoPredictedService", "SASP-NoVirtualQueue",
             "SASP-NoCriticalCoverage", "SASP-GreedyAssignment"]
ABL_LABELS = ["No service-aware score", "No virtual queue", "No coverage safeguard", "Greedy assignment"]
SUITES = ["stress", "stationary", "long_stress", "scale12", "scale16", "scale24", "scale32", "ablation"]
PHASES = ["pre_surge", "surge", "degradation", "recovery"]
G = "cohort_goodput_mbps"
O = "queue_overflow_packet_probability"
E = "deadline_expiration_class0"
REPORT_METRICS = [G, O, E, "deadline_expiration_class1", "deadline_expiration_class2",
                  "energy_j", "total_packet_failure_probability", "throughput_mbps",
                  "controller_median_ms", "controller_p95_ms"]
CAPTIONS = {
    "Fig02_v810_main_comparison": "Performance under the short-stress evaluation.",
    "Fig03_v810_stress_robustness": "Phase-wise behavior and performance across operating regimes.",
    "Fig04_v810_ablation": "Paired performance changes under SASP component ablations.",
    "Fig05_v810_scalability_cost": "Performance and controller cost across device/channel configurations.",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class Source:
    """Read archive members directly; do not extract arbitrary paths."""
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        require(self.path.exists(), f"Input does not exist: {self.path}")
        self.archive = zipfile.ZipFile(self.path) if self.path.is_file() and zipfile.is_zipfile(self.path) else None
        require(self.archive is not None or self.path.is_dir(), f"Expected directory or ZIP: {self.path}")
        if self.archive:
            self.names = {}
            for n in self.archive.namelist():
                if n.endswith(("/", "\\")):
                    continue
                normalized = n.replace("\\", "/")
                require(normalized not in self.names, f"Duplicate archive path: {normalized}")
                self.names[normalized] = n
        else:
            self.names = {p.relative_to(self.path).as_posix(): p for p in self.path.rglob("*") if p.is_file()}

    def read(self, name):
        return self.archive.read(self.names[name]) if self.archive else self.names[name].read_bytes()

    def json(self, name):
        return json.loads(self.read(name).decode("utf-8-sig"))


def finite_mean(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def confidence(values):
    values = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    require(len(values) >= 2, "At least two evaluable blocks are required for an interval.")
    avg = float(values.mean())
    half = float(student_t.ppf(.975, len(values)-1) * values.std(ddof=1) / np.sqrt(len(values)))
    return dict(mean=avg, ci95_low=avg-half, ci95_high=avg+half, n_blocks=len(values))


def load_inputs(paths):
    sources = [Source(p) for p in paths]
    runs, protocols = {}, {}
    for source in sources:
        for name in sorted(source.names):
            if name.rsplit("/", 1)[-1] == "protocol.json":
                protocol = source.json(name)
                if str(protocol.get("version", "")).startswith("8.10"):
                    protocols[digest(protocol)] = protocol
            if name.rsplit("/", 1)[-1] != "run_manifest.json":
                continue
            spec = source.json(name).get("specification", {})
            if spec.get("role") != "final" or spec.get("suite") not in SUITES:
                continue
            require(str(spec.get("version", "")).startswith("8.10"), f"Not v8.10: {source.path}:{name}")
            suite = spec["suite"]
            require(suite not in runs, f"Multiple final {suite} runs found. Supply only the intended run set.")
            prefix = name[:-len("run_manifest.json")]
            require(prefix+"completion.json" in source.names, f"Incomplete suite: {suite}")
            completion = source.json(prefix+"completion.json")
            run_hash = digest(spec)
            require(completion["run_sha256"] == run_hash, f"Completion hash mismatch: {suite}")
            expected_methods = {"SASP", *ABLATIONS} if suite == "ablation" else set(METHODS)
            require(set(spec["methods"]) == expected_methods, f"Unexpected method set: {suite}")
            records, paired = {}, defaultdict(list)
            for episode_name in sorted(source.names):
                if not episode_name.startswith(prefix+"episodes/") or not episode_name.endswith(".json"):
                    continue
                envelope = source.json(episode_name)
                r = envelope["record"]
                require(envelope["record_sha256"] == digest(r), f"Record hash mismatch: {episode_name}")
                require(r["run_sha256"] == run_hash and r["suite"] == suite, f"Wrong run: {episode_name}")
                require(r["method"] in expected_methods, f"Unknown method: {episode_name}")
                b, i = r["block"], r["index"]
                require(0 <= b < spec["blocks"] and 0 <= i < spec["scenarios_per_block"], f"Invalid scenario index: {episode_name}")
                expected_seed = 410_000_000 + SUITES.index(suite)*1_000_000 + b*1000+i
                require(r["scenario_seed"] == expected_seed, f"Unexpected seed: {episode_name}")
                key = (r["method"], b, i)
                require(key not in records, f"Duplicate episode: {episode_name}")
                m = r["metrics"]
                require(m["monitored_cohort_complete"] == 1, f"Incomplete monitored cohort: {episode_name}")
                for field in ("monitored_residual_packets", "packet_outcome_conservation_error", "reservation_conflict_rate"):
                    require(m[field] == 0, f"Failed {field}: {episode_name}")
                require(m["offered_packets"] == sum(m[k] for k in ["completed_on_time_packets", "queue_overflow_packets", "deadline_expired_packets"]), f"Packet counts do not balance: {episode_name}")
                # The full-cohort undefined-class handling is already part of v8.10.
                for c in (0, 1, 2):
                    require(m[f"offered_packets_class{c}"] > 0 or m[f"deadline_expiration_class{c}"] is None,
                            f"Undefined full-cohort class rate was stored as zero: {episode_name}")
                records[key] = m
                paired[(b, i)].append((r["method"], r["exogenous_sha256"]))
            expected = spec["blocks"] * spec["scenarios_per_block"] * len(expected_methods)
            require(len(records) == expected == completion["episodes"], f"Missing episodes: {suite}, found {len(records)}, expected {expected}")
            for key, group in paired.items():
                require({g[0] for g in group} == expected_methods and len({g[1] for g in group}) == 1,
                        f"Common-random-number matching failed: {suite}, {key}")
            runs[suite] = dict(spec=spec, records=records, source=str(source.path), prefix=prefix)
            if prefix+"summary.csv" in source.names:
                runs[suite]["original_summary"] = list(csv.DictReader(io.StringIO(source.read(prefix+"summary.csv").decode("utf-8-sig"))))
            print(f"Validated {suite}: {len(records)} episodes", flush=True)
    require(set(runs) == set(SUITES), "Missing final suites: "+", ".join(sorted(set(SUITES)-set(runs))))
    hashes = {run["spec"]["protocol_sha256"] for run in runs.values()}
    require(len(hashes) == 1, "Runs do not use one frozen protocol.")
    protocol_hash = next(iter(hashes))
    require(protocol_hash in protocols, "Matching protocol.json not found. Include the outputs directory or main-results ZIP.")
    protocol = protocols[protocol_hash]
    require(protocol.get("confirmatory_eligible") is True, "Protocol is not eligible for final evaluation.")
    for suite, run in runs.items():
        spec, plan = run["spec"], protocol["final_plan"][suite]
        require(spec["source_hashes"] == protocol["source_hashes"], f"Source identity mismatch: {suite}")
        require(spec["dpp_v"] == protocol["selected_v"], f"DPP setting mismatch: {suite}")
        require(spec["blocks"] == plan["blocks"] and spec["scenarios_per_block"] == plan["scenarios"]
                and set(spec["methods"]) == set(plan["methods"]), f"Frozen plan mismatch: {suite}")
        run["config"] = protocol["environments"][suite]
        for row in run.get("original_summary", []):
            check = stats(run, row["method"], row["metric"])
            for k in ("mean", "ci95_low", "ci95_high"):
                require(np.isclose(check[k], float(row[k]), rtol=1e-8, atol=1e-11), f"Summary mismatch: {suite}/{row['method']}/{row['metric']}")
    for source in sources:
        if source.archive:
            source.archive.close()
    return runs, protocol, protocol_hash


def stats(run, method, metric, scale=1.):
    block_means, valid = [], 0
    spec = run["spec"]
    for b in range(spec["blocks"]):
        values = [run["records"][method, b, i][metric] for i in range(spec["scenarios_per_block"])]
        valid += sum(v is not None and np.isfinite(v) for v in values)
        mean = finite_mean(values)
        block_means.append(None if mean is None else mean * scale)
    return {**confidence(block_means), "n_evaluable_scenarios": valid}


def paired_stats(run, method, metric, scale, direction):
    block_means, valid = [], 0
    spec = run["spec"]
    for b in range(spec["blocks"]):
        values = []
        for i in range(spec["scenarios_per_block"]):
            full = run["records"]["SASP", b, i][metric]
            ablated = run["records"][method, b, i][metric]
            if full is not None and ablated is not None:
                values.append(direction * (ablated-full) * scale)
        valid += len(values)
        block_means.append(finite_mean(values))
    return {**confidence(block_means), "n_evaluable_scenarios": valid}


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Figures:
    def __init__(self, runs, output, dpi):
        self.runs, self.output, self.dpi, self.values = runs, output, dpi, []
        plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8,
            "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.linewidth": .65,
            "lines.linewidth": 1.2, "lines.markersize": 4, "pdf.fonttype": 42,
            "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})

    def value(self, fig, panel, suite, method, metric, scale=1., unit="", direction=None):
        r = self.runs[suite]
        result = stats(r, method, metric, scale) if direction is None else paired_stats(r, method, metric, scale, direction)
        self.values.append(dict(figure=fig, panel=panel, suite=suite, method=method,
            metric=metric, unit=unit, estimand="mean" if direction is None else ("SASP_minus_ablation" if direction == -1 else "ablation_minus_SASP"), **result))
        return result

    @staticmethod
    def error(s):
        return np.array([[s["mean"]-s["ci95_low"]], [s["ci95_high"]-s["mean"]]])

    @staticmethod
    def decorate(ax, title, ylabel=None):
        ax.set_title(title, loc="left", pad=7, fontweight="bold")
        if ylabel:
            ax.set_ylabel(ylabel, labelpad=4)
        ax.grid(axis="y", alpha=.20, linewidth=.5)
        ax.set_axisbelow(True)

    @staticmethod
    def legend(fig, y=.995):
        handles = [Line2D([], [], color=COLORS[m], marker=MARKERS[m], label=LABELS[m]) for m in METHODS]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, y),
                   ncol=6, frameon=False, columnspacing=1., handlelength=1.45, handletextpad=.4)

    @staticmethod
    def target(ax, y, label):
        ax.axhline(y, color=".45", ls=":", lw=.8, zorder=0)
        ax.text(.03, .97, label+" (dotted)", transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color=".35")

    def save(self, fig, stem):
        fig.savefig(self.output/(stem+".pdf"), metadata={"Title": CAPTIONS[stem], "Creator": VERSION})
        fig.savefig(self.output/(stem+".png"), dpi=self.dpi)
        plt.close(fig)
        (self.output/(stem+"_caption.txt")).write_text(CAPTIONS[stem]+"\n", encoding="utf-8")

    def main_comparison(self):
        fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.60), sharey=True)
        fig.subplots_adjust(left=.125, right=.985, bottom=.20, top=.82, wspace=.29)
        for j, (metric, scale, title, unit) in enumerate([(G, 1, "(a) Goodput", "Mbps"), (O, 100, "(b) Overflow", "%")]):
            for y, m in enumerate(METHODS):
                s = self.value("2", chr(97+j), "stress", m, metric, scale, unit)
                axes[j].errorbar(s["mean"], y, xerr=self.error(s), fmt=MARKERS[m], color=COLORS[m], capsize=2)
            self.decorate(axes[j], title)
            axes[j].set_xlabel(f"Cohort goodput ({unit})" if j == 0 else f"Overflow ({unit})")
            axes[j].grid(axis="x", alpha=.2)
        axes[1].axvline(3, color=".45", ls=":", lw=.8)
        axes[1].text(3, .98, "3% target", transform=axes[1].get_xaxis_transform(), ha="right", va="top", rotation=90, fontsize=6.5)
        for y, m in enumerate(METHODS):
            for c, marker, shift in [(0, "o", -.13), (1, "^", .13)]:
                s = self.value("2", "c", "stress", m, f"deadline_expiration_class{c}", 100, "%")
                axes[2].errorbar(s["mean"], y+shift, xerr=self.error(s), fmt=marker, color=COLORS[m], capsize=2, markersize=3.7)
        self.decorate(axes[2], "(c) Critical deadlines")
        axes[2].set_xlabel("Expiration (%)")
        axes[2].grid(axis="x", alpha=.2)
        axes[2].axvline(4, color=".55", ls=":", lw=.7)
        axes[2].axvline(5, color=".55", ls="--", lw=.7)
        fig.legend(handles=[Line2D([], [], marker="o", color=".3", ls="", label="Emergency (target: 4%)"),
                            Line2D([], [], marker="^", color=".3", ls="", label="Medical (target: 5%)")],
                   loc="upper center", ncol=2, frameon=False)
        axes[0].set_yticks(range(6), [LABELS[m] for m in METHODS])
        axes[0].set_ylim(5.6, -.6)
        self.save(fig, "Fig02_v810_main_comparison")

    def stress_robustness(self):
        fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.95))
        fig.subplots_adjust(left=.08, right=.985, bottom=.12, top=.82, wspace=.39, hspace=.78)
        self.legend(fig)
        fig.text(.53, .902, "Short-stress phases", ha="center", fontsize=8, fontweight="bold")
        fig.text(.53, .448, "Operating regimes", ha="center", fontsize=8, fontweight="bold")
        top = [("throughput_mbps", 1, "(a) Served throughput", "Mbps"),
               ("queue_overflow_packet_probability", 100, "(b) Overflow", "%"),
               ("deadline_expiration_class0_by_arrival_cohort", 100, "(c) Emergency expiration", "%")]
        bottom = [(G, 1, "(d) Cohort goodput", "Mbps"), (O, 100, "(e) Overflow", "%"), (E, 100, "(f) Emergency expiration", "%")]
        for row, definitions in enumerate([top, bottom]):
            for col, (metric, scale, title, unit) in enumerate(definitions):
                ax = axes[row, col]
                categories = PHASES if row == 0 else ["stationary", "stress", "long_stress"]
                for mi, m in enumerate(METHODS):
                    vals = [self.value("3", chr(97+row*3+col), "stress" if row == 0 else cat,
                                      m, f"phase_{cat}_{metric}" if row == 0 else metric, scale, unit) for cat in categories]
                    xx = np.arange(len(categories)) + (mi-2.5)*.019
                    ax.errorbar(xx, [s["mean"] for s in vals], yerr=np.concatenate([self.error(s) for s in vals], axis=1),
                                color=COLORS[m], marker=MARKERS[m], capsize=1.5, lw=1.1 if m != "SASP" else 1.6)
                self.decorate(ax, title, unit)
                ax.set_xticks(range(len(categories)), ["Pre-surge", "Surge", "Degradation", "Recovery"] if row == 0
                              else ["Stationary", "Short stress", "Long stress"], rotation=23, ha="right")
                ax.set_xlim(-.20, len(categories)-.80)
                if col == 1:
                    self.target(ax, 3, "3% target")
                if col == 2:
                    self.target(ax, 4, "4% target")
        self.save(fig, "Fig03_v810_stress_robustness")

    def ablation(self):
        fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.55), sharey=True)
        fig.subplots_adjust(left=.195, right=.985, bottom=.25, top=.81, wspace=.32)
        fig.text(.59, .945, "Positive values indicate deterioration; paired 95% intervals", ha="center", fontsize=7.5)
        definitions = [(G, 1, -1, "(a) Goodput loss", "Mbps", .05),
                       (O, 100, 1, "(b) Overflow increase", "pp", .1),
                       (E, 100, 1, "(c) Emergency expiration", "pp", .1)]
        for col, (metric, scale, direction, title, unit, threshold) in enumerate(definitions):
            ax = axes[col]
            for y, method in enumerate(ABLATIONS):
                s = self.value("4", chr(97+col), "ablation", method, metric, scale, unit, direction)
                ax.errorbar(s["mean"], y, xerr=self.error(s), fmt="o", color=["#0072B2", "#E69F00", "#009E73", "#CC79A7"][y], capsize=2)
            ax.axvline(0, color=".4", lw=.8, ls=":")
            ax.set_xscale("symlog", linthresh=threshold, linscale=1)
            ax.margins(x=.15)
            ax.autoscale_view()
            ticks = [-.1, 0, .1, 1, 10] if col > 0 else [0, .05, .5, 5]
            lo, hi = ax.get_xlim()
            ticks = [t for t in ticks if lo <= t <= hi]
            ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
            ax.set_xlabel(f"Change ({unit}; symlog)")
            self.decorate(ax, title)
            ax.set_title(title, loc="left", fontsize=8)
            ax.grid(axis="x", alpha=.2)
        axes[0].set_yticks(range(4), ABL_LABELS)
        axes[0].set_ylim(3.5, -.5)
        self.save(fig, "Fig04_v810_ablation")

    def scalability(self):
        fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.65))
        fig.subplots_adjust(left=.075, right=.985, bottom=.22, top=.77, wspace=.37)
        self.legend(fig)
        suites = ["scale12", "scale16", "scale24", "scale32"]
        labels = [f"{self.runs[s]['config']['n_devices']}/{self.runs[s]['config']['n_channels']}" for s in suites]
        for j, (metric, scale, title, unit) in enumerate([(G, 1, "(a) Cohort goodput", "Mbps"),
                (O, 100, "(b) Overflow", "%"), ("controller_median_ms", 1, "(c) Controller median", "ms")]):
            ax = axes[j]
            for mi, m in enumerate(METHODS):
                vals = [self.value("5", chr(97+j), s, m, metric, scale, unit) for s in suites]
                ax.errorbar(np.arange(4)+(mi-2.5)*.019, [s["mean"] for s in vals], yerr=np.concatenate([self.error(s) for s in vals], axis=1),
                            color=COLORS[m], marker=MARKERS[m], capsize=1.5, lw=1.1 if m != "SASP" else 1.6)
            self.decorate(ax, title, unit)
            ax.set_xticks(range(4), labels)
            ax.set_xlabel("Devices/channels")
            ax.set_xlim(-.2, 3.2)
            if j == 1:
                self.target(ax, 3, "3% target")
            if j == 2:
                slot_ms = self.runs[suites[0]]["config"]["slot_s"]*1000
                self.target(ax, slot_ms, f"{slot_ms:g} ms slot")
        self.save(fig, "Fig05_v810_scalability_cost")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", nargs="+", default=["outputs"], help="Directories or ZIPs containing all eight final suites and protocol.json (default: outputs)")
    parser.add_argument("--output-dir", type=Path, default=Path("publication_figures_v8_10"))
    parser.add_argument("--figure1", type=Path, help="Optional existing system-model PDF, copied unchanged")
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution; PDF remains vector (default: 300)")
    args = parser.parse_args(argv)
    require(72 <= args.dpi <= 1200, "PNG dpi must be between 72 and 1200.")
    if args.figure1:
        require(args.figure1.is_file() and args.figure1.suffix.lower() == ".pdf", "--figure1 must point to an existing PDF")
    runs, protocol, protocol_hash = load_inputs(args.inputs)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    plotter = Figures(runs, out, args.dpi)
    plotter.main_comparison()
    plotter.stress_robustness()
    plotter.ablation()
    plotter.scalability()
    write_csv(out/"plot_source_values_v8_10.csv", plotter.values)
    table = []
    for suite, run in runs.items():
        for method in run["spec"]["methods"]:
            for metric in REPORT_METRICS:
                table.append(dict(suite=suite, method=method, metric=metric, **stats(run, method, metric)))
    write_csv(out/"all_metrics_v8_10.csv", table)
    figure1_note = "Figure 1 is external and was not copied."
    if args.figure1:
        dst = out/"Fig01_SASP_SystemModel.pdf"
        if args.figure1.resolve() != dst:
            shutil.copyfile(args.figure1, dst)
        figure1_note = "Figure 1 copied unchanged; its scientific content was not validated by this plotting script."
    latex = ["% Place each block near its first citation; panel details belong in the text.",
             "% Copy the PDFs beside your main .tex file, or adjust the paths.",
             "% Keep your existing Figure 1 block before these blocks."]
    for stem, caption in CAPTIONS.items():
        latex.extend([r"\begin{figure*}[!t]", r"    \centering",
                      r"    \includegraphics[width=\textwidth]{"+stem+".pdf}",
                      r"    \caption{"+caption+"}", r"    \label{fig:"+stem+"}", r"\end{figure*}", ""])
    (out/"figure_blocks_v8_10.tex").write_text("\n".join(latex), encoding="utf-8")
    manifest = dict(plotter_version=VERSION, protocol_sha256=protocol_hash, selected_dpp_v=protocol["selected_v"],
        figure1=figure1_note, generated_figures=[s+".pdf" for s in CAPTIONS],
        suites={s: dict(source=r["source"], member_prefix=r["prefix"], episodes=len(r["records"]),
                       blocks=r["spec"]["blocks"], scenarios_per_block=r["spec"]["scenarios_per_block"],
                       run_sha256=digest(r["spec"])) for s, r in runs.items()},
        checks="PASS: frozen plan, record/run hashes, complete cohorts, conservation, paired exogenous hashes, supplied summary means and intervals",
        limitations=["This is a plotting input check, not an independent simulator or baseline correctness audit.",
            "Support suites and phase breakdowns are exploratory; displayed 95% t intervals are not multiplicity-adjusted.",
            "Phase rates preserve the exporter max(arrivals,1) denominator convention; see FIGURE_NOTES.md.",
            "Approximate t intervals are not clipped at zero; symlog ablation axes preserve signs and zeros."],
        runtime=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__),
        plotting_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/"publication_figure_manifest_v8_10.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    notes = Path(__file__).with_name("FIGURE_NOTES.md")
    if notes.exists() and notes.resolve() != (out/notes.name):
        shutil.copyfile(notes, out/notes.name)
    print(f"Generated Figures 2--5, source values, captions and LaTeX blocks in {out}")
    print(figure1_note)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
