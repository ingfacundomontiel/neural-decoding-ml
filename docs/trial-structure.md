# Trial structure and virtual corridor geometry

This document captures what we currently understand about the physical /
virtual layout of the experimental rig and the structure of a single trial.
It is the reference for any analysis that needs to interpret position
values or trial boundaries in physical terms.

> **All numeric values here are estimates / working assumptions** based on
> the lab's prior experiments and discussion with collaborators
> (2026-06-12). They should be confirmed against the lab's official rig
> documentation before any thesis-grade claim is made.

> **For the bin-level preprocessing pipeline** (immobility +
> trial-transition filtering) that uses these landmarks, see
> [`preprocessing.md`](./preprocessing.md).

## Position units

Positions in the `.mat` recordings are stored in **wheel-encoder degrees**.
Conversion to corridor centimetres uses the factor used by the lab's prior
experiments:

  `position_cm = position_degrees × 0.1222`

Source of the factor: variable `cm_per_degree` in
`marinburginlab-labPrograms/.../san/all-datasets-plot-predictions.py`. The
same constant is exported by `analysis/loaders.py` as `CM_PER_DEGREE` to
keep code and docs in sync.

> **Confirmed (2026-07-14).** The `0.1222` factor was verified with the
> director and is correct. This is no longer a working assumption.

## Virtual corridor layout

| Landmark | Position (cm) | Notes |
|---|---|---|
| Trial start | 0 | Animal teleports here at the start of each trial. |
| Context entry | 44.7 | Rewarded or non-rewarded context begins (visual cue). Confirmed 2026-08-26. |
| Context exit | 62.9 | Animal exits the context. Confirmed 2026-08-26. |
| Reward zone | starts at 62.9 | Begins at the context exit; width not documented. Water delivered if `rewCtxt = 1 AND rewOdor = 1`. |
| Natural corridor end | not documented | Until the lab confirms it, the context exit is the last known landmark; everything past it is treated as post-context (reward zone / licking drift / running past the end). |

> **History.** The 2026-06-12 working estimates were 42 / 59 / 60 cm. They were
> replaced on 2026-08-26 by the values above (context entry / exit confirmed
> with the director). In raw encoder units: entry ≈ 366°, exit ≈ 515°.

## Trial flow

1. Animal teleports to position 0 at the start of the trial.
2. Animal runs forward; the wheel encoder increments `position`.
3. At 44.7 cm, the rewarded / non-rewarded context begins.
4. The odor is delivered during the context window. The variables
   `rewCtxt` and `rewOdor` in the `.mat` are **time-varying flags** that
   fire only while the stimulus is active — they are not constant across
   the whole trial.
5. At 62.9 cm the animal exits the context and reaches the reward zone. If
   `rewCtxt = 1 AND rewOdor = 1`, water is delivered.
6. The animal either continues running (no reward, or quickly past the
   reward zone) or stops to lick the water.
7. The trial ends and the animal teleports back to position 0.

## How a trial actually ends

Two distinct termination paths, depending on whether the animal licks
the reward:

- **Rewarded + licked (`rewCtxt = 1`, `rewOdor = 1`, `goResponse = 1`).**
  The rig extends the corridor while licking continues. The teleport
  happens only after the animal stops licking. The trial duration in
  these cases is variable and licking-dependent — see "Known drift" below.

- **Any other trial (no reward delivered, or rewarded with no lick).**
  The rig uses a **fixed time-out** measured from the moment the animal
  crosses the natural corridor end. After that timeout elapses, the
  animal is teleported back to position 0 regardless of what it is doing.

The exact timeout duration is not documented by the lab, but empirically
estimated as **≈ 2.2 s** (see below). This value should be confirmed
against the rig's official spec.

### Empirical timeout estimate

For every clean non-rewarded, no-lick trial, the delay between the last
bin at `position < 60 cm` and the bin where the trial ends
(`trialFinalBin`) was measured. The estimate was validated across four
expert datasets covering both regions (piriform and hippocampus):

| Dataset | n trials | median delay | p25 | p75 |
|---|---|---|---|---|
| `Pir_L5_expert_bins200ms` | 89 | 2.20 s | 2.20 | 2.20 |
| `Pir_L3_expert_bins200ms` | 187 | 2.20 s | 2.20 | 2.20 |
| `CA3_S18_expert_bins200ms` | 161 | 2.20 s | 2.20 | 2.40 |
| `DG_S19_expert_bins200ms` | 152 | 2.20 s | 2.20 | 2.40 |

The median is identical across datasets and the interquartile range is
near zero — strong evidence that the timeout is a property of the rig,
not of the animal or session.

Two further checks support this reading:

1. *Sanity*: rewarded trials in which the animal happened not to lick
   (`rewCtxt = 1`, `rewOdor = 1`, `goResponse = 0`) — rare but present
   in `Pir_L5` — show the same median delay (2.30 s, n = 12).
   Consistent: with no licking, the rig falls back to the same
   timeout regardless of whether the trial was rewarded.
2. *Control*: rewarded + licked trials show median delays of
   6.6–10.6 s depending on the dataset, clearly different from the
   2.2 s of the no-lick case. Consistent with the corridor extending
   while licking.

The full analysis is in `exploration/01d-non-rewarded-trial-delay.py`.

The estimate is robust against the choice of "corridor end" threshold:
medians shift by ≤ 0.6 s as the threshold moves between 50 and 70 cm,
which is consistent with a running animal traversing those few extra
centimetres before the rig timer expires. And the timer is reported in
**seconds**, so it does not depend on the unresolved
position-to-cm conversion factor.

> **Open question** about how this interacts with the current
> preprocessing pipeline: the immobility filter
> (`analysis/preprocessing.py`) currently marks only a 1 s window
> before each `trialFinalBin` as trial-transition padding. The remaining
> ~1.2 s of the rig timeout is not filtered explicitly. Whether those
> bins should also be removed is a decision to take with the lab — they
> may carry locomotion signal worth keeping, or they may be uninformative
> tail that contaminates the decoder.

## Known drift in rewarded + licked trials

When the animal stops at the reward zone to lick the water, the rig is
**configured to extend the virtual corridor while the animal continues
licking**. The teleport back to position 0 happens only once the animal
stops licking. As a consequence, in rewarded-context + rewarded-odor
trials where the animal licks, the recorded `position` can exceed the
natural corridor end and the trial can last considerably longer than a
non-rewarded trial.

This is a deliberate rig behaviour, not a recording artefact.

## Out-of-corridor session-start artefact (DG-naive)

Separately from the legitimate drift above, the three **DG-naive** recordings
(`DG_S38`, `DG_S39`, `DG_S41`) each contain a single malformed **first trial**:
the trial never terminates for several minutes and the encoder keeps
accumulating position far past the corridor — up to **26508°** (~3240 cm) in
`DG_S41`, versus a valid trial that never exceeds **~1200°** (~147 cm) anywhere
in the project. Confirmed with the director (2026-07-14) as a session-start
recording artefact, not real behaviour.

The distinction that matters: a *valid* long trial (the animal stops and stays
still) is long in **duration** but its **position stays bounded** within the
corridor. This artefact is long in duration **and** its position leaves the
corridor entirely, because the animal keeps running while the trial-termination
logic is inactive. Duration alone therefore cannot separate the two — **peak
position** can.

Left unfiltered, this single trial holds 13–21 % of all bins with positions
40× the corridor scale, which destroys the global standardisation of the
position target and collapses the position decoder (test R² as low as −47 for
`DG_S41`). It is removed in preprocessing by the **out-of-corridor trial
filter** (`thPosition`, default 1500° ≈ 183 cm) — see
[`preprocessing.md`](./preprocessing.md) and the decision record in
[`specs/decisions.md`](../specs/decisions.md). Every other dataset is left
untouched by this filter.

## Constants in code

The landmarks above are exported as module-level constants in
`analysis/loaders.py`, alongside `CM_PER_DEGREE`:

- `CONTEXT_START_CM = 44.7`
- `CONTEXT_END_CM = 62.9`
- `REWARD_START_CM = CONTEXT_END_CM`
- `CORRIDOR_END_CM = CONTEXT_END_CM` — working assumption while the natural
  corridor end is undocumented (see the layout table above).

`analysis/loaders.py` is the single source of truth for these values. If
any landmark is later corrected against the lab's official rig spec,
update the constant there and amend the corresponding row in this
document with the new value and date.

## Open questions to confirm with the director

- Exact natural corridor length (not documented; `CORRIDOR_END_CM` currently
  falls back to the context exit).
- Exact width of the reward zone.
- Whether the `rewCtxt` / `rewOdor` flags overlap exactly with the visible
  context window, or have lead/lag relative to it.

(The `cm_per_degree` calibration factor, `0.1222`, was confirmed with the
director on 2026-07-14 — see [Position units](#position-units) above.)
