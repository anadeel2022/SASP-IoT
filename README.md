# SASP: Service-Aware Spectrum-Power Scheduling for Reliable IoT Access

Code and simulation results accompanying **Service-Aware Spectrum–Power Scheduling With Causal Virtual Queues for Reliable IoT Access**, by Muhammad Faisal Siddiqui, Adeel Iqbal, Sung Won Kim, and Mohammed Al-Naeem.

SASP is a training-free scheduler for heterogeneous IoT uplinks with finite queues, deadlines, uncertain channels, and discrete powers. This release contains the **v8.10 evaluation** against MaxRateMatching, MaxWeight, WeightedMaxWeight, earliest-deadline-first (EDF), and LyapunovDPP-Energy. It does not train or compare learned policies. No publication DOI or acceptance is asserted here.

## Contents

| Path | Contents |
|---|---|
| `simulation/` | Six original Python files, byte-identical to the frozen protocol |
| `simulation/tests/` | 25 existing tests of objectives, assignment, deadlines, cohort accounting, randomness, and safeguards |
| `plotting/publication_plots_v8_10.py` | Publication plotter, revision `8.10-plots-2` |
| `data/protocol.json` | Original frozen protocol; selected DPP coefficient V=0.1 |
| `data/validation_selection.json` | Original validation selection record |
| `data/final_*/` | Original scenario metrics, block summaries, and paired comparisons for eight final suites |
| `data/plot_source_values_v8_10.csv` | Plotted points and intervals in display units |
| `data/all_metrics_v8_10.csv` | Additional metrics, including energy and runtime |
| `figures/` | Archived vector PDFs for Figures 2–5 |
| `docs/FIGURE_NOTES.md` | Estimands, figure interpretation, and limitations |
| `docs/VERIFICATION.md` | Repository preparation checks |
| `CITATION.cff` | Software citation metadata |

Figure 1 is a separately prepared system diagram. Manuscript and administrative submission files are not included.

## Installation and tests

Python 3.12 is recommended. No GPU, PyTorch, pandas, or LaTeX installation is required. `requirements.txt` supplies dependency compatibility ranges, not an exact environment lock. Recorded experiments used Python 3.12.10, NumPy 2.5.3, and SciPy 1.18.1 on Windows 11; timing is machine-dependent.

From the repository root in PowerShell:

```powershell
python -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
Push-Location simulation
& "..\.venv\Scripts\python.exe" -m unittest discover -s tests -v
Pop-Location
```

Linux/macOS equivalent:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
(cd simulation && ../.venv/bin/python -m unittest discover -s tests -v)
```

## Smoke test

From the root:

```powershell
& ".\.venv\Scripts\python.exe" simulation/experiment_v8_10.py run --protocol data/protocol.json --role pilot --suite stress --blocks 1 --scenarios 1 --output outputs/pilot_stress
```

This executes six methods on one pilot scenario; it is an execution check, not publication evidence. For Linux/macOS, replace the PowerShell executable prefix with `.venv/bin/python`.

The runner hashes every top-level `.py` file in `simulation/`. Keep plotting scripts and helpers outside that directory. `.gitattributes` preserves frozen source and evidence bytes across platforms.

## Re-run the frozen experiments

```powershell
foreach ($suite in @("stress", "stationary", "long_stress", "scale12", "scale16", "scale24", "scale32", "ablation")) {
    & ".\.venv\Scripts\python.exe" simulation/experiment_v8_10.py run --protocol data/protocol.json --role final --suite $suite --output "outputs/final_$suite"
    if ($LASTEXITCODE -ne 0) { throw "Experiment failed: $suite" }
}
```

Main suites use 10 blocks of 72 matched scenarios; support suites use five blocks of 24. There are 12,840 method-scenario records: 4,320 in each main suite, 600 in ablation, and 720 in each other supporting suite. Full experiments are substantial runs; estimate runtime using the pilot.

Repeat the identical command and output directory to resume. Completed episodes are checked and skipped. Changed source, runtime, or configurations are rejected during resume; use fresh output directories for another environment. Do not run concurrent writers in the same directory.

## Generate Figures 2–5

After all eight suites complete:

```powershell
& ".\.venv\Scripts\python.exe" plotting/publication_plots_v8_10.py --inputs outputs data --output-dir publication_figures
```

`outputs` supplies the newly generated episodes and manifests; `data` supplies the frozen protocol. The plotter validates integrity and exports PDFs, PNGs, and source-value CSVs.

**Archived CSVs alone are not plotter inputs.** Detailed episode JSONs and runtime manifests are omitted from this compact repository. Inspect the archived PDFs and plotted-value CSV immediately, or run the experiments above to regenerate complete outputs. Do not fabricate manifests or bypass integrity checks.

If you separately possess the original result ZIPs, they can be plotted directly:

```powershell
& ".\.venv\Scripts\python.exe" plotting/publication_plots_v8_10.py --inputs v8_10_main_results_20260923_141308.zip final_.zip --output-dir publication_figures
```

These ZIPs are not bundled here; their checksums are in `data/ARCHIVE_PROVENANCE.json`.

## New validation or modified methods

For a separate experimental revision:

```powershell
& ".\.venv\Scripts\python.exe" simulation/experiment_v8_10.py validate --output outputs/new_validation
& ".\.venv\Scripts\python.exe" simulation/experiment_v8_10.py freeze --validation outputs/new_validation/validation_selection.json --output outputs/new_protocol.json
```

Validation selects DPP's energy coefficient from {0, 0.1, 1, 10, 100}; it does not tune SASP. The freeze records source hashes and refuses to overwrite a protocol. Changed-source or retuned evaluations are a new study, not the archived manuscript revision.

## Interpretation

Cohort goodput counts complete on-time payload from measurement-window arrivals, including eligible follow-up completions, divided by the measurement-window duration. It differs from served-bit throughput. Overflow and expiration use offered-packet denominators; warm-start packets are excluded.

Main comparisons use paired block inference and Holm correction; intervals are unadjusted. Supporting suites are exploratory. Read `docs/FIGURE_NOTES.md` for phase exporter conventions and undefined class rates.

Archived short-stress SASP results are 4.176 Mbps goodput and 1.644% overflow. Improvements over EDF accompany higher Emergency expiration and energy consumption. Long-stress overflow exceeds the target, and controller runtimes exceed the simulated 1-ms slot duration. Computation delay is not inserted into queue dynamics. The simulations do not demonstrate real-time deployment feasibility.

SASP's coefficients are fixed design parameters inherited from pre-confirmatory development and retained unchanged for revised baselines. No systematic search for their original values is documented in this release.

## Citation and contact

Use `CITATION.cff` and identify the Git commit and frozen protocol hash when reporting a reproduction. Add manuscript publication details only when available.

Corresponding author: Adeel Iqbal, School of Computer Science and Engineering, Yeungnam University; adeeliqbal@yu.ac.kr.
