# SAN 2026 poster — plan

Living planning document for the poster presented at the annual meeting of the
Sociedad Argentina de Investigación en Neurociencias (SAN), 2026. Decisions are
dated; open topics are listed at the end and must be closed before the
corresponding section is built.

## Format constraints (set by SAN)

| Constraint | Value |
|---|---|
| Maximum dimensions | 90 cm wide × 125 cm tall |
| Orientation | Portrait (vertical) — landscape is not accepted |
| Material | Paper |
| Language | English |

## Production

- Authored as an HTML page, exported to PDF for printing.

## Reference posters

- `conferences/san-2025/` (SVG) — primary reference. Portrait, 90 × 120 cm
  (from its viewBox, 2551.5 × 3401.25 pt). Canva export: all text is outlined
  (no `<text>` elements), 14 embedded bitmaps — regions can be cropped as vector.
- `conferences/sfn-2025/` (PDF) — secondary reference. Landscape.

## Source abstract

`conferences/san-2026/Abstract SAN 2026.md` — the accepted abstract. It fixes the
title, the author list and affiliations, and the claims the poster has to
support:

1. Spatial location is encoded in the PCx once animals learn the task, not before.
2. Spatial encoding precision increases around behaviourally relevant locations.
3. PCx and hippocampal decoding performances are comparable.

Claim 2 depends on open topic T1.

## Decisions (2026-09-26)

### Global

- **Title, authors, affiliations**: exactly as in the abstract (including the
  equal-contribution mark). Funding: as in the previous posters (to confirm the
  exact grant list).
- **Page size**: designed at the maximum allowed, 90 × 125 cm.
- **Results pipeline**: the final one — position-tail filter (515° ≈ 62.9 cm,
  D-017) + Gaussian smoothing of spike counts (σ = 200 ms) + no history window
  (current bin only, 0/1/0). Position: Batch 007; velocity (if kept): Batch 008
  — same bins and folds as position. Results trees: `results_notail_smooth_w010/`.
- **Datasets** (changed 2026-09-27 from all 17): the **14 recordings presented at
  SfN 2025** (12 mice; S17 and S18 were recorded in both CA3 and DG). Pir_N59,
  DG_S24 and DG_S41 are left out (`EXCLUDED_STEMS` in `scripts/poster_data.py`).

  | Region | Expert | Naive | Total |
  |---|---|---|---:|
  | Piriform cortex | L3, L5, N31 | L7, N53 | 5 |
  | CA3 | S17, S18 | S30, S31 | 4 |
  | DG | S17, S18, S19 | S38, S39 | 5 |
  | **Total** | **8** | **6** | **14** |

- **Group statistics**: none. With 2 naive recordings in piriform, results are
  shown per recording (5 folds each) and described, not tested.

### Per section

1. **Introduction** — layout and images kept from SAN 2025; text rewritten (the
   2025 text claims feed-forward *and recurrent* networks and speed decoding as
   a result — no longer true).
2. **Task** — kept; graphics recropped from the 2025 SVG as vector and the text
   labels re-set in HTML (fixes the overlapping "odours and inhalation
   detection" and the broken "odour A or B" labels). If the result looks poor,
   the original assets will be retrieved.
3. **Prior evidence (Federman et al., 2024; Poo et al., 2021)** — kept, with
   the position axis of the *context decoding* panel corrected: the published cm
   values are inflated (~1.9×) by a degree-to-cm conversion error, confirmed by a
   co-author of the paper; the correct scale is this project's
   (`CM_PER_DEGREE = 0.1222`, context 44.7–62.9 cm). Caption states the panel is
   adapted with a corrected position axis. Exact factor to calibrate on the
   panel's own context edges (same physical landmarks: 366° / 515°) and confirm.
4. **Sample data + preprocessing** — test a single merged figure: population
   raster on top (original data), position and velocity below with the removed
   bins shaded by cause (trial transition/teleport, immobility, post-context
   tail) and the context band marked. Corrections to the current
   `02-data-structure-figure` output: replace the stale "corridor end (60 cm)"
   line by the context band (44.7–62.9 cm); add velocity units. The d′ and
   firing-rate plots are dropped; neuron and trial quality control (low firing
   rate < 0.1 Hz, unstable neurons, low-d′ trials, out-of-corridor artifact
   trials) are listed as text with counts. Use "immobility", never "low
   activity". Decide after seeing the test figure.
5. **Decoder (MLP)** — diagram drawn from the actual network of the final
   pipeline (verified in `external/ibioml/ibioml/models.py`,
   `experiments/run_experiment.py`, and the `best_config_per_fold` of the 17
   Batch 007 studies):
   - Input: the smoothed spike counts of the N kept neurons in the current
     200 ms bin, plus a 2-unit one-hot of the trial's context (`rewCtxt`) —
     `input_size = N + 2`. Inputs z-scored with train-fold statistics.
   - Hidden: 1–3 layers of 128–512 units, each Linear → BatchNorm → ReLU →
     Dropout (0–0.5). Selected per fold: 3 layers in 48/85 folds, 2 in 31/85.
   - Output: one **linear** unit (position, z-scored target, de-standardised
     to cm for evaluation). One separate model per target.
   - Training: Adam, MSE loss, batch 32, ≤ 200 epochs, early stopping
     (patience 20). Hyperparameters per outer fold by Optuna (TPE, 20 trials).
6. **Cross-validation** — trial-wise, 5 outer folds, a single train/validation
   hold-out inside each (not nested 5×5). Colours switched to neutrals; linked to
   the boxplots ("5 test folds → 5 points per box").
7. **Datasets** — neurons vs trials scatter, 17 points, colour = region, fill =
   cohort, direct animal labels, poster style. Second scatter (position R² vs
   speed R²): to confirm, depends on T2.
8. **Decoding performance** — R² boxplots of the 17 recordings grouped by
   region, median printed on each box, no legend boxes.
9. **Error along the corridor** — the section the poster is built around.
   Content depends on T1. The rewarded/unrewarded split of the SfN poster is
   dropped (inside the corridor the per-trial-type curves are indistinguishable,
   `exploration/10-error-by-trial-type.py`).
10. **Predictions over real data** — regenerated from the final pipeline; one box
    per region with a region label on the frame (SfN style); identical plot
    heights and axis scales across regions; predicted trace in the region colour;
    the time axis labelled as concatenated test bins (removed bins are absent).
11. **Conclusions** — rewritten from the new numbers (e.g. naive CA3 ≈ naive
    piriform, so "in naive animals only CA3 decodes" no longer holds).

### Style guide (approved)

- **Production rule**: every figure is generated at its final physical size
  (matplotlib `figsize` = its slot on the poster) and inserted at 100 % as SVG.
  One shared `poster.mplstyle` (typeface, sizes, line widths, palette).
- **Typeface**: one family everywhere (HTML and figures): IBM Plex Sans (must
  be installed locally for matplotlib).
- **Type scale** (printed size): title 100–110 pt · authors 48 pt ·
  affiliations 28 pt · box titles 56–60 pt · panel titles 36–40 pt · body
  28 pt (min 24) · axis labels 26 pt · ticks and legends 22 pt · numbers
  inside plots 24 pt · references 22 pt. Hard floor: 20 pt.
- **Semantic palette** (each colour means one thing; derived from Okabe–Ito):
  - Region = hue: piriform `#0B63A5`, CA3 `#8E3B8F`, DG `#C0581B`. Naive =
    same hue, lighter / hollow markers / dashed lines.
  - Green reserved for the rewarded context (the VR's green wall), grey for the
    unrewarded one.
  - Context zone: pale yellow band `#FFF3C4` on every position axis.
  - Real data: ink `#1E1E1E`; predictions: region colour.
  - Cross-validation diagram: neutrals only.
- **Boxes**: white background, rounded corners, 3 pt dark-grey border, no drop
  shadow.
- **Result box titles are claims** (as box 3 of SAN 2025).
- **Grid**: 2 columns of ~41 cm; fine layout budgeted in cm once content is
  locked.

## Open topics (discuss in depth before locking)

- **T1 — Regression toward the training mean in the error-by-position
  analysis.** An MSE-trained regressor shrinks uncertain predictions toward
  the mean of its training targets, so its error is lowest near that mean and
  highest at the edges of the training range even if the neural information is
  uniform. With the tail included, the mean fell near the context zone; in the
  posend515 pilot, removing the tail moved the context to the edge of the range
  and its error rose (+2 / +6 cm). This directly affects abstract claim 2.
  To discuss: the mechanism, how to separate information from shrinkage
  (position-matched chance reference, other controls), which analyses/runs to
  approve, and what the section can honestly claim. The existing 08 figures are
  from the old pipeline; none exist yet for the final one.
- **T2 — Velocity as a control variable.** Proposed framing: speed decoding is a
  positive control (the pipeline reads behaviour from these neurons; a naive
  recording that decodes speed but not position shows a position-specific
  deficit), while speed is also the main confound for position (see T3). To
  decide: whether velocity stays, and in which panels (position-vs-speed R²
  scatter, speed boxplot, speed traces).
- **T3 — Decoupling control ("is it just speed?").** How the behaviour-only
  control works, what it does and does not rule out, and whether it is
  included. The existing results (`exploration/14-…`) are from the old
  pipeline and would need recomputing on the final one (CPU only).
- **T4 — Runs and analyses to approve**, after T1–T3.
- **QR code** — proposed target: the public IBioML repository
  (github.com/mariburginlab-labPrograms/IBioML), with a contact e-mail next to
  it. To approve.

## Progress log

### 2026-09-26 (afternoon, autonomous session)

**Infrastructure** (all under `conferences/san-2026/`, uncommitted):
`scripts/poster.mplstyle` + `scripts/poster_style.py` (palette, cm sizing,
IBM Plex Sans registered from `assets/fonts/`), `scripts/poster_data.py`
(final-pipeline loader, kept-bin reconstruction exact-verified against the
pickles). Figures land in `figures/` (SVG + PNG preview). PDF export verified:
Chrome `--print-to-pdf` honours `@page { size: 90cm 125cm }` (one page,
89.99 × 124.98 cm).

**Figures generated** (provisional slot sizes; re-run after the layout budget):
`fig_summary.py` → `datasets_neurons_vs_trials`, `boxplot_r2_pos`,
`boxplot_r2_vel`, `scatter_r2_pos_vs_speed`; `fig_predictions.py` →
`pred_{pos,vel}_{Pir,CA3,DG}` (best recording per region × cohort, fold closest
to its median; same axes everywhere); `fig_preprocessing.py` →
`preprocessing_N31_97-100` (test of the merged sample-data + preprocessing
figure). Poster-level key (region colours, hollow/light = naive) replaces
per-figure legends.

**Assets**: `assets/san2025/` — vector-rendered crops of the 2025 poster (brain
schematic, histology, task graphics, prior-evidence panels, logos) at ~300 dpi
of their final size. **Federman et al. (2024) axis factor measured**: the
context edges of the published panel sit at 83.9 and 118.1 cm vs 44.7 / 62.9 cm
→ factor 1.877 / 1.878 (≈ 1.88); corrected ticks = published position / 1.88.

**T1 preliminary** (`scripts/t1_error_by_position.py`, `figures/t1/`):
- Simulation with the MSE-optimal decoder and *uniform* information reproduces
  the artefact: MAE is U-shaped with its minimum at the training mean; with the
  tail removed the context zone sits at the edge of the range and its error
  rises; with the tail kept the minimum moves onto the context. The local slope
  of predicted vs true position stays flat under shrinkage; extra information
  inside the context produces a slope peak and an MAE dip at the context entry.
- Real data (final pipeline): MAE curves follow the no-information V (|x − μ|)
  in mid-corridor, so raw MAE cannot locate information. Per recording, the
  predicted-position jump at the context entry (mean prediction at 47–55 cm minus
  at 35–43 cm) is large in **all** piriform recordings, naive included
  (7–13 cm; hippocampus 0.4–8 cm) — consistent with an odour-onset response.
  What separates expert from naive piriform is the graded tracking *before* the
  context (slope 5–40 cm: experts 0.27–0.34 except N59 0.06; naive 0.06–0.09)
  and a lower in-context MAE (10–14 vs 18–21 cm).
- New-pipeline versions of the 08 heatmaps: `figures/t1/08-final-pipeline/`.

**T3 preliminary** (`scripts/t3_behavior_preview.py`, `figures/t3/`, gradient-
boosted trees, same bins and test folds):
- A non-linear behaviour-only decoder (speed ±400 ms + context) reaches R²
  comparable to the neural MLP in several recordings (e.g. Pir L3 0.46 vs 0.46,
  CA3 S18 0.48 vs 0.44, DG S19 0.43 vs 0.25). The ridge control of
  `exploration/14-…` (R² 0.11–0.19) underestimated the behavioural ceiling.
- Incremental test (same learner): adding the neurons to speed + context improves
  R² in **all 17** recordings (Δ +0.04…+0.28, 4–5/5 folds each). In expert
  piriform the gain concentrates inside the context zone (MAE 18.1 → 8.6 cm N31,
  13.9 → 8.7 L3, 19.3 → 12.6 L5); before the context it is ~1–2 cm.

**GPU** (see `docs/experiment-log.md`): Batch 011 (behaviour-only MLP) and
Batch 012 (neural + behaviour MLP), 17 datasets each, two runners per batch,
launched 13:38; Batch 011 starts after Batch 010. Observed ~0.85 min per Optuna
trial with three concurrent runners (≈ 1.3× slower than one) → both expected to
finish between the early morning and mid-morning of 2026-09-27.

**Poster skeleton** `poster.html` (preview: `figures/_preview_poster_v1.png`):
header, the three kept boxes (task labels re-set and old borders removed:
`assets/san2025/task_graphics_fixed.png`; Federman axis corrected:
`prior_evidence_corrected.png`), methods box with the merged preprocessing figure
and the MLP / CV diagrams (`scripts/fig_diagrams.py`, inline SVG so text uses the
page font), poster-level key, datasets + position-vs-speed scatters, position
boxplot, predictions per region. T1/T3 section and conclusions are placeholders.
**Density check** (content laid out without clipping): left column ends at
~136 cm and right at ~131 cm against a 122.5 cm limit — ~14 cm over on the left
(before the diagrams were added) and ~9 cm over on the right *without* the T1/T3
section (~20 cm) and the conclusions (~10 cm). Roughly 30–40 cm must be cut; see
open topic T5.

- **T5 — Space budget.** Candidate cuts: two-line title and one-line authors
  (~4 cm); box titles 48 pt and shorter claims (~3 cm per column); drop the
  bottom row of box 3 (decoding vs number of neurons, ~9 cm); preprocessing
  figure 26 → 20 cm tall; prediction rows 9 → 7 cm or fewer regions; datasets
  scatter replaced by a one-line summary or merged with the speed scatter.

### 2026-09-27

- **Decision**: none of the new analyses (T1 shrinkage / local slope, T3
  behaviour-only and incremental decoders) go on the poster — too early to present.
  The poster uses the results already established.
- Batches 011 and 012 finished (17/17 each; headlines in `docs/experiment-log.md`):
  the behaviour-only MLP matches the neural MLP (median Δ R² −0.005) and adding the
  neurons improves decoding in 16/17 recordings (median +0.116).

- **Content decisions (closing T2, T5 and the corridor section):**
  - Corridor section: heatmap P(decoded | true) + MAE per position, best expert per
    region, with the best naive MAE of the same region dashed
    (`scripts/fig_corridor.py`). Caption makes no zone-precision claim and notes that
    squared-error decoders err least near the mean training position.
  - Velocity: only the position-vs-speed R² scatter (datasets box). Caption: speed
    is decoded from every recording; in naive piriform cortex position is not — no
    claim that this rules out a speed confound.
  - Box 3: Federman et al. top row (odour + context decoding, corrected axis) and
    the Poo et al. decoding-vs-neurons panel; the Federman bottom row and the plus
    maze drawing are dropped.
  - Wording: decoding language throughout ("position can be decoded"), not
    "encodes a spatial map".
- **Layout fits** (`scripts/measure.sh`; after the 14-recording switch and 4 pt box
  borders): left column ends at 119.0 cm, right at 122.4 cm (limit 122.5). Title 72 pt (2 lines), authors 38 pt (1 line), box titles
  46 pt, body 30 pt. `poster-san-2026-montiel.pdf`: one page, 90.0 × 125.0 cm, 4.3 MB.
- Acknowledgments as a one-line footer in the former bottom margin (columns
  untouched; lowest ink 1.1 cm above the paper edge). Grant list taken from the
  SfN 2025 poster — to confirm.
- Still to review: conclusions draft (with the director), intro text, box titles
  (e.g. "After learning: piriform cortex ≈ hippocampus"), acknowledgments/funding
  line, QR code.

## Follow-ups (after the congress)

- **Bayesian decoder with a uniform prior** (Poisson rate maps from the raw counts
  of `data/processed_notail_raw_w010/`, same folds, MAP estimate and the posterior
  probability at the true position, per position bin): a decoder without the
  implicit occupancy prior of the MSE-trained MLP, to answer T1 ("where along the
  corridor is position encoded?") free of the shrinkage artefact.
- Write up T1 / T3 (simulation, local slope, behaviour-only and incremental
  decoders) as thesis material; confidence intervals over trials still missing.

## To review with the director

- The rewritten conclusions (claims 1–3 against the final-pipeline numbers).
- The exact degree-to-cm factor behind the published Federman et al. (2024)
  axis.
