# Trial structure and virtual corridor geometry

This document captures what we currently understand about the physical /
virtual layout of the experimental rig and the structure of a single trial.
It is the reference for any analysis that needs to interpret position
values or trial boundaries in physical terms.

> **All numeric values here are estimates / working assumptions** based on
> the lab's prior experiments and discussion with collaborators
> (2026-06-12). They should be confirmed against the lab's official rig
> documentation before any thesis-grade claim is made.

## Position units

Positions in the `.mat` recordings are stored in **wheel-encoder degrees**.
Conversion to corridor centimetres uses the factor used by the lab's prior
experiments:

  `position_cm = position_degrees × 0.1222`

Source of the factor: variable `cm_per_degree` in
`marinburginlab-labPrograms/.../san/all-datasets-plot-predictions.py`. The
same constant is exported by `analysis/loaders.py` as `CM_PER_DEGREE` to
keep code and docs in sync.

## Virtual corridor layout

| Landmark | Position (cm) | Notes |
|---|---|---|
| Trial start | 0 | Animal teleports here at the start of each trial. |
| Context entry | ~42 | Rewarded or non-rewarded context begins (visual cue). |
| Context exit | ~59 | Animal exits the context. |
| Reward zone | ~59 → 60 | 1–2 cm wide; water delivered if `rewCtxt = 1 AND rewOdor = 1`. |
| Natural corridor end | ~60–65 | Nominal end of the virtual corridor visible to the animal. |

## Trial flow

1. Animal teleports to position 0 at the start of the trial.
2. Animal runs forward; the wheel encoder increments `position`.
3. Around ~42 cm, the rewarded / non-rewarded context begins.
4. The odor is delivered during the context window. The variables
   `rewCtxt` and `rewOdor` in the `.mat` are **time-varying flags** that
   fire only while the stimulus is active — they are not constant across
   the whole trial.
5. Around ~59–60 cm, the animal reaches the reward zone. If
   `rewCtxt = 1 AND rewOdor = 1`, water is delivered.
6. The animal either continues running (no reward, or quickly past the
   reward zone) or stops to lick the water.
7. The trial ends and the animal teleports back to position 0.

## Known drift in rewarded + licked trials

When the animal stops at the reward zone to lick the water, the rig is
**configured to extend the virtual corridor while the animal continues
licking**. The teleport back to position 0 happens only once the animal
stops licking. As a consequence, in rewarded-context + rewarded-odor
trials where the animal licks, the recorded `position` can exceed the
natural corridor end and the trial can last considerably longer than a
non-rewarded trial.

This is a deliberate rig behaviour, not a recording artefact.

## Constants in code

The landmarks above are exported as module-level constants in
`analysis/loaders.py`, alongside `CM_PER_DEGREE`:

- `CONTEXT_START_CM = 42.0`
- `CONTEXT_END_CM = 59.0`
- `REWARD_END_CM = 60.0`
- `CORRIDOR_END_CM = 60.0`

`analysis/loaders.py` is the single source of truth for these values. If
any landmark is later corrected against the lab's official rig spec,
update the constant there and amend the corresponding row in this
document with the new value and date.

## Open questions to confirm with the director

- Exact natural corridor length (currently estimated 60–65 cm).
- Exact width of the reward zone (currently 1–2 cm).
- Exact `cm_per_degree` calibration factor (currently 0.1222).
- Whether the `rewCtxt` / `rewOdor` flags overlap exactly with the visible
  context window, or have lead/lag relative to it.
