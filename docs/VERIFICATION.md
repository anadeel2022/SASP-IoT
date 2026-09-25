# Repository verification

Prepared September 25, 2026.

All six Python source files in `simulation/` exactly match the SHA-256 values in the original frozen protocol. The source code was copied without implementation changes. The plotting script is stored separately because the experiment runner hashes all top-level Python files in its own directory.

All 25 existing tests passed in the packaged layout. Tests cover scheduling objectives against exhaustive search, matching, feasibility, deadline boundaries, packet accounting, common randomness, validation rules, and resume safeguards.

A one-block, one-scenario pilot using `data/protocol.json` completed for all six primary methods from the repository root. This is a packaging smoke test, not a new final evaluation.

The archived CSVs contain 12,840 method-scenario records across eight suites. Packet-outcome conservation was checked for every row: offered packets equal overflow plus on-time completions plus expiration. Scenario CSVs, summaries, paired comparisons, protocol, and validation-selection bytes were preserved from their original archives. `data/ARCHIVE_PROVENANCE.json` identifies those archives and record counts.

The original episode JSON traces and per-run manifests are not included. Consequently this package permits inspection and analysis of archived scenario metrics but cannot re-run the original episode-level integrity audit using CSVs alone. Full reruns generate the required records for the unmodified plotter.

Packaging verification environment: Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, Matplotlib 3.10.8, Linux. This differs from the manuscript timing machine. No claim of identical timing across environments is made.

Dependency bounds are compatibility ranges. The full confirmatory simulations were not rerun during repository preparation. Original manuscript results and timings remain archived evidence.
