# Figure definitions and manuscript integration

## Scope and method labels

The plots use only the completed v8.10 baseline-repair evaluations. They do not
reuse old MAPPO, MAPPO-RS or PPO-Lagrangian diagnostics. The legend maps MaxRate
to MaxRateMatching, W-MaxWeight to WeightedMaxWeight, and DPP-Energy to
LyapunovDPP-Energy. EDF, MaxWeight and SASP retain their stored names.

Figure 1 is external. Figures 2--5 use a 7.16-inch canvas and are intended for
full-width placement in the two-column manuscript. This is a design choice,
not a statement of a current journal requirement. Captions are deliberately
short; describe the definitions below in the methods/results text.

## Aggregation and uncertainty

For each suite, method and metric, the script averages finite scenario values
within each block, then gives equal weight to evaluable block means. Missing
full-cohort class rates, stored as null when that class has no offered packets,
are excluded rather than replaced by zero. The export includes the number of
evaluable scenarios and blocks for each point. The main supplied evaluations
use 10 blocks of 72 scenarios, while the supplied support evaluations use
5 blocks of 24 scenarios. The script reads these counts from the frozen plan.

The displayed interval is the approximate, unadjusted Student-t interval:

```latex
\bar{x} \pm t_{0.975,B-1}\frac{s_b}{\sqrt{B}}.
```

The ablation intervals use within-scenario paired differences, averaged within
blocks before the same interval calculation. These are not confidence intervals
obtained by subtracting independently estimated interval bounds. Phase breakdowns,
long stress, scalability and ablations are exploratory. There are no significance
stars; consult the original multiplicity-adjusted `paired_comparisons.csv` files
for the planned comparisons. The script checks summary means and intervals but
does not revalidate those p-values.

Approximate intervals may extend below zero for rare nonnegative outcomes. The
figures retain those bounds and the source CSV stores them unchanged. Such bounds
are a limitation of this approximation, not negative physical failure rates.
An observed zero does not establish zero underlying risk. No positive floor is
substituted for zero on logarithmic axes.

## Figure 2: main comparison

Goodput is the complete on-time payload of packets arriving during the measured
horizon, including their eligible follow-up completions, divided by the measured
horizon duration. It differs from served-bit throughput during the horizon:

```latex
G_{\mathrm{cohort}}=
\frac{\sum_{p\in\mathcal{A}_{H}} L_p\,
\mathbf{1}\{p\text{ completed on time}\}}{H\tau}.
```

The unit plotted is Mbps. Queue overflow and class deadline expiration use
offered-packet denominators. Deadline expiration excludes immediate queue-overflow
losses, so the two metrics must not be treated as interchangeable total failure
probabilities. Figure 2(c) shows Emergency (circles, 4% target) and Medical
(triangles, 5% target); the vertical lines use dotted and dashed styles respectively.
Colour identifies the method. Figure 2 plots the short-stress suite.

Keep energy in the main comparison table using `all_metrics_v8_10.csv`.
The supplied results support a goodput/overflow advantage for SASP over EDF
under short stress, accompanied by higher energy and worse Emergency expiration.
Do not describe SASP as uniformly superior across all metrics.

## Figure 3: phase behavior and operating regimes

The top row uses all saved short-stress scenario metrics. Panel (a) is served-bit
throughput during each phase. Panels (b,c) are arrival-phase cohort loss ratios,
with delayed outcomes attributed to the packet's arrival phase. In particular,
panel (c) is NOT the number of expirations occurring in that phase divided by
new arrivals in that phase. It reads the stored
`phase_*_deadline_expiration_class0_by_arrival_cohort` fields.

The v8.10 environment exports phase ratios with denominator `max(arrivals, 1)`.
Consequently an empty arrival-phase/class contributes an exported zero. Complete
phase arrival counts are not retained in the non-trace records, and the plotter
does not infer or invent them. Phase plots preserve this convention exactly;
unlike the full-cohort class means, they must not be described as conditional
means restricted to phases with at least one class arrival. Future experiments
can eliminate this reporting limitation by exporting phase offered counts and
null for undefined class rates. Full-cohort Figure 2/3-bottom/4/5 results are
unaffected by this phase-specific exporter limitation.

The bottom row compares the stationary, short-stress and long-stress suites.
The supplied measured horizons are 180, 180 and 900 slots, respectively, with
the configured follow-up. Panel (d) is cohort goodput, not the top row's served-bit
throughput. Connecting lines guide the eye across categorical conditions; the
suites have distinct scenario seeds and are not paired across regimes.
Do not describe these results as learned-policy transfer or as a controlled
one-factor comparison of horizon length. Preserve the long-stress target
violations in the discussion. Failure of the tested methods to meet a target
does not prove that target is mathematically infeasible.

## Figure 4: paired ablations

All panels use the separately evaluated ablation suite. Positive values indicate
deterioration, with goodput loss = full SASP minus ablation and probability
increase = ablation minus full SASP. Probability differences are in percentage
points (pp), not relative percentages.

```latex
\Delta G=G_{\mathrm{SASP}}-G_{\mathrm{abl}},\qquad
\Delta p=100\bigl(p_{\mathrm{abl}}-p_{\mathrm{SASP}}\bigr).
```

The axes use symmetric logarithmic scaling, linear near zero, to retain visibility
of both small and large effects without dropping the dominant service-score
ablation. The linear thresholds are 0.05 Mbps, 0.1 pp and 0.1 pp. Values and
intervals are transformed only for display; source CSV entries stay in original
units. A negative value means improvement on that metric.

`No service-aware score` maps to `SASP-NoPredictedService`: it removes the combined
service-aware score components (CQI, normalized service, completion and critical
completion bonus). It is not an isolated prediction-error experiment.
`No coverage safeguard` maps to `SASP-NoCriticalCoverage`: it removes the
post-assignment safeguard, not the critical scoring term. `No virtual queue`
and `Greedy assignment` map to the correspondingly named stored variants.
The supplied corrected comparisons do not establish a contribution from the
coverage safeguard on the four primary endpoints. Do not imply that every
component provides a statistically established benefit.

## Figure 5: scalability and controller cost

The x-axis reports both device and channel counts (N/C). Both change across
configurations, so these are not fixed-resource scaling experiments.
Panel (c), labelled `Controller median`, averages scenario-level median controller times using the same block
aggregation. These are measured controller selection plus virtual-queue update
times, excluding simulator stepping, audit/IO overhead and the first five
primary slots. They are not overall simulator wall time or a pooled per-slot
median. The 1-ms reference is the configured slot duration, not a demonstrated
real-time deadline guarantee. Include the runtime limitations in the manuscript.
Mean scenario-level p95 controller times and their block intervals are retained
in `all_metrics_v8_10.csv` for the main text or a supporting table.

## Placement

Use the generated `figure_blocks_v8_10.tex` blocks near their first textual
references with `figure*[!t]` and `width=\textwidth`. Keep the system diagram
before them so these become Figures 2--5. Update old Figure 6 references to
Figure 3(d)--(f). The previous manuscript's v8.9 numbers, captions and learned
diagnostics must not be carried over unchanged. Final page savings depend on
the revised text and float placement and require recompiling the manuscript.
