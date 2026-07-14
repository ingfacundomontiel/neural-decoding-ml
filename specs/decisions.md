# Decisions — Architecture & Tooling

Append-only record. Every non-trivial decision has: date, the choice, the
alternatives considered, and why the choice wins. Format is lightweight ADR.

---

## D-001 — Rebuild inside `neural-decoding-ml/` (don't create a fourth directory)

**Date:** 2026-04-17
**Status:** accepted

**Context.** Three sibling directories already exist for the same project. A fourth
would add noise and split attention further.

**Decision.** Reuse `neural-decoding-ml/`. Archive current contents to `_legacy/`,
keep the git history, rebuild fresh in the same repo.

**Alternatives.**
- New directory `thesis-neural-decoding/` — clean slate, but more clutter on disk.
- Delete `neural-decoding-ml/` and start fresh — loses git history and the
  documented OOP strategy that might still inform design.

**Why.** The name already describes the project. Git history preserves the
stalled-rewrite journey (useful for the thesis retrospective). `_legacy/` keeps
reference material one folder away without polluting the active workspace.

---

## D-002 — Fork `ibioml` via git submodule (option A)

**Date:** 2026-04-17
**Status:** accepted

**Context.** `ibioml` is a lab-shared library that will continue to evolve, with
at least one known bug (index-type issue in `preprocess_data`) already
monkey-patched in production scripts. Facundo is an author on the upstream repo.

**Decision.** Fork `mariburginlab-labPrograms/IBioML` to `facundomontiel/IBioML`
on GitHub, add it as a submodule at `external/ibioml/`. Upstream remote on the
fork lets us rebase/cherry-pick improvements. Contributions flow back via PRs
from the fork.

**Alternatives.**
- In-tree vendoring (`src/ibioml/`) — simplest but makes upstream diffs painful.
- Depend on PyPI `ibioml` — can't fix bugs fast, can't audit source cleanly.

**Why.** Clean boundary between "thesis code" and "library I'm contributing to."
The submodule pointer is visible in git, so thesis reviewers can see exactly
which `ibioml` commit backs every result. Matches the dual role Facundo has
(user + editor).

**Cost.** Submodule discipline — must remember to update/commit the submodule
pointer when the fork advances. Acceptable.

---

## D-003 — `uv` as the package/environment manager

**Date:** 2026-04-17
**Status:** accepted

**Context.** Current workflow is plain `venv/` + `requirements.txt`. Not enough
for thesis-grade reproducibility.

**Decision.** `uv` with `pyproject.toml` and `uv.lock`.

**Alternatives.**
- Plain `venv` + `pip-tools` — workable but more manual, slower resolves.
- `poetry` — solid, but slower than `uv`, and has had governance/tooling churn.
- `conda`/`mamba` — overkill, slow, overlapping scope with system package managers.

**Why.** `uv` has: fastest resolver on the market, native `pyproject.toml`,
proper lock file, trivial install (`curl -LsSf https://astral.sh/uv/install.sh | sh`),
Python version management built in, active development by Astral (the `ruff`
team). No need for a second tool for linting/formatting later — `ruff` integrates.

---

## D-004 — Python 3.11, pinned in `.python-version`

**Date:** 2026-04-17
**Status:** accepted

**Why 3.11.** Broadly supported by every ML library we use (PyTorch, scikit-learn,
scipy, numpy). Not so new that compatibility breaks (some scientific libraries
still lag on 3.12/3.13). Long-term support window covers thesis submission.
Can bump later without friction.

---

## D-005 — Jupytext-paired notebooks for exploration

**Date:** 2026-04-17
**Status:** accepted

**Context.** Notebooks are natural for narrative + inline figures, terrible for
diffs and code review.

**Decision.** Every notebook is paired with a `.py` file in percent format via
Jupytext. Source of truth is the `.py`. `.ipynb` is regenerated and typically
gitignored (or stripped of outputs on commit).

**Alternatives.**
- Plain `.ipynb` — bad git diffs, merge conflicts on cell metadata.
- Plain `.py` scripts only — loses narrative + inline figure integration.
- Marimo — interesting reactive alternative, but less familiar to lab users,
  not the thesis battleground.

**Why.** Best of both worlds. Diffs are clean. Thesis narrative can still embed
figures from the notebooks. Convertible back to `.ipynb` on demand.

**Convention.** Commit the `.py`. Gitignore the `.ipynb` unless it contains
essential output figures (case-by-case).

---

## D-006 — Plain Markdown for documentation, no MkDocs (yet)

**Date:** 2026-04-17
**Status:** accepted

**Why.** The project is personal for now, possibly lab-shared later. MkDocs adds
tooling cost without clear benefit at this stage. Every doc renders natively on
GitHub. If/when the project goes public, migrating Markdown → MkDocs is trivial
(matches what `ibioml` already does).

---

## D-007 — English everywhere

**Date:** 2026-04-17
**Status:** accepted

**Why.** Thesis will be evaluated by an international committee, `ibioml` is
English, and mixing languages creates friction. Existing Spanish docs (e.g.,
`docs/current-status.md`, `docs/data-structure.md` in the legacy project) move
to `_legacy/` untouched.

---

## D-008 — Move legacy contents to `_legacy/` inside the same repo

**Date:** 2026-04-17
**Status:** superseded by D-014

**Context.** Old code has reference value (the OOP strategy, bug notes,
existing trainers) but shouldn't be in the active workspace.

**Decision.** Everything currently at `neural-decoding-ml/` root (except `.git`,
`.gitignore`, and the new `specs/`) moves into `_legacy/`. Tracked in git so
it's discoverable but never imported.

**Alternatives.**
- Delete and rely on git history — works, but slower to revisit during the thesis.
- Separate archive repo — overkill.

**Why.** Zero-friction retrieval when writing thesis "what I tried before" sections.
Can be deleted later when truly no longer needed.

---

## D-009 — No `data/processed/` in git, only `data/raw/`

**Date:** 2026-04-17
**Status:** superseded by D-013

**Why.** Raw `.mat` files are the canonical ground truth and not huge (few MB each).
Processed pickles are derivable from raw + code + preprocessing params, so they're
reproducible artifacts. Gitignored to keep the repo lean. `data/raw/README.md`
documents every raw file; future `data/processed/README.md` documents how each
processed file was produced.

---

## D-010 — One `.mat` file per dataset, latest version wins

**Date:** 2026-04-17
**Status:** accepted

**Context.** Same `.mat` files appear in multiple directories with different
modification dates.

**Decision.** For each dataset, keep only the most recently modified copy. If two
copies differ in content (not just metadata), flag in `dataset-inventory.md`
and defer to Facundo before choosing. `marinburginlab/.../sfn/datasets/` is the
default authoritative location.

---

## D-011 — No new experiments until exploration + audit are done

**Date:** 2026-04-17
**Status:** accepted

**Why.** Facundo's current priority is understanding the data more deeply and
auditing `ibioml`. Running new experiments now would obscure that. Experiments
directory is scaffolded but empty until Phase 4.

---

## D-012 — Fixes to `ibioml` follow test-first discipline

**Date:** 2026-04-17
**Status:** accepted

**Why.** Upstream `ibioml` may not have comprehensive tests. Every fix we
contribute adds a regression test in `external/ibioml/tests/`. Protects the
thesis results from silent breakage if the library is edited by others, and
raises the bar for upstream quality.

---

## D-013 — No datasets committed to git (supersedes D-009)

**Date:** 2026-06-12
**Status:** accepted

**Context.** D-009 originally accepted committing `data/raw/` to git on the
grounds that the `.mat` files are small. After confirming with the lab
director that the recordings are unpublished lab data, that choice was
reconsidered: pushing them to GitHub — even a private repo, let alone the
public one this repository lives in — copies lab data to third-party
infrastructure under a personal account. That data is not Facundo's to
redistribute.

**Decision.** No datasets in git, at any path:

- `*.mat` is in the project `.gitignore`.
- `data/raw/` and `data/processed/` are documented in their own READMEs but
  their contents are not tracked.
- Existing `.mat` blobs were removed from history via `git filter-repo` at
  the time of this decision.
- The canonical source of every `.mat` is documented in
  [`dataset-inventory.md`](dataset-inventory.md) and
  [`../data/raw/README.md`](../data/raw/README.md). To reproduce the dataset
  layout on a new machine, copy from the lab paths listed there.

**Alternatives.**
- **Keep them in git, private repo** — still puts lab data on third-party
  infrastructure under a personal account. Rejected.
- **Git LFS** — same data-residency issue plus extra tooling. Rejected for
  this thesis; could revisit if the lab decides centrally that data lives
  in LFS under the lab org.
- **Separate private repo for data** — defers the data-residency question
  rather than solving it.

**Why.** Lab data should remain on lab-controlled infrastructure until
publication. The thesis code stays reproducible on any machine via the
inventory + README + access to the lab source paths.

**Cost.** New machines need an extra step: copy the `.mat` files from the
lab sources following `data/raw/README.md`. Acceptable.

---

## D-014 — `_legacy/` directory kept locally, not committed (supersedes D-008)

**Date:** 2026-06-12
**Status:** accepted

**Context.** D-008 accepted moving the previous workspace contents into
`_legacy/` and tracking that directory in git. After the discovery (during
the same push that produced D-013) that `_legacy/processed-datasets/`
contained derivative data from lab recordings — preprocessed pickles,
trained model weights, and experiment outputs — the same data-residency
reasoning that drove D-013 applies here. Pushing those artifacts to a
public GitHub repo under a personal account is not appropriate.

**Decision.** The `_legacy/` directory stays on disk for local reference
(the original D-008 motivation — "zero-friction retrieval when writing
the thesis retrospective"), but its contents are not tracked:

- `.gitignore` excludes `_legacy/` entirely.
- Data extensions are also gitignored globally as defence-in-depth:
  `*.pickle`, `*.pkl`, `*.pth`, `*.npy`, `*.npz`, `*.h5`, `*.hdf5`.
- The previously tracked `_legacy/*` blobs were removed from git history
  via `git filter-repo` at the time of this decision.
- The original lab sources of any derivable artifact remain accessible
  at `marinburginlab-labPrograms/.../experiments/`, as already noted by
  D-013.

**Why.** Same rationale as D-013: unpublished lab data — *including
artifacts derived from it* (pickles, weights, results JSONs) — should
not be redistributed via GitHub. The local copy still serves the
original D-008 purpose.

**Cost.** A new clone won't have `_legacy/`. To rebuild it, the developer
needs access to the original workspace or the lab paths. Acceptable.

---

## D-015 — Bump `ibioml` submodule pin to `upstream/main` (supersedes audit-baseline pin from D-002)

**Date:** 2026-07-05
**Status:** accepted

**Context.** D-002 (2026-04-17) fixed the submodule pin at
`thesis/audit-baseline` (commit `4e477c3`) as an anchor for Phase 1's
audit reproducibility. During Phase 2 the audit findings that mattered
for downstream work were contributed upstream via PR #9 (preprocess
filters, merged 2026-07-03) and PR #10 (rewOdor load, setup URL,
`__version__`, merged 2026-07-05). Both PRs are now on
`mariburginlab-labPrograms/IBioML` `main` at commit `acbbf3e`.

Continuing to train models against `4e477c3` would mean using a
`preprocess_data` that:

- has neither `periods_immobility` nor `clean_unstable_neurons`;
- still drops `rewOdor` (loads `rewCtxt` twice);
- reports `__version__ == "0.1.2"` regardless of the installed release.

**Decision.** Bump the submodule pin from `4e477c3` (`thesis/audit-baseline`)
to `acbbf3e` (`upstream/main`). Along with the bump:

- Retire `analysis/preprocessing.py`. It was a verbatim copy of
  `periods_immobility` used only because the pin blocked importing
  directly from the lib. Its own module docstring anticipated this
  retirement.
- Update `exploration/01c-immobility-filter.py:43` to import
  `periods_immobility` from `ibioml.preprocess_data` (same signature,
  now shipped upstream).

`analysis/loaders.py` stays — it ships lab-specific `.mat` constants
(corridor geometry, `CM_PER_DEGREE`) that don't live in the lib.

**Alternatives.**

- Cherry-pick individual fixes into `thesis/audit-baseline` — duplicates
  work every time upstream advances, requires maintaining two divergent
  branches on the fork.
- Move to `pip install ibioml @ git+...` without a submodule — simpler
  in requirements, but loses the ability to edit/debug the lib in-place
  on the GPU host.

**Why.** The audit-baseline anchor served its purpose: audit findings
are now upstream, and `docs/ibioml-audit.md` still cites `4e477c3` for
provenance. Going forward, the thesis tracks `upstream/main` and
re-bumps as the lib advances. Reviewers can still walk the submodule
commit graph to see the exact `ibioml` state behind each result — the
property D-002 was really protecting.

---

## D-016 — Out-of-corridor trial filter (position-based, not duration-based)

**Date:** 2026-07-14
**Status:** accepted

**Context.** The batch MLP run (17 datasets × dual target) produced a
catastrophic position decoder for `DG_S41_naive` (mean test R² ≈ −47; folds to
−181), with `DG_S38_naive` (−1.1) and `DG_S39_naive` (−0.16) also negative.
Root cause: each of the three DG-naive recordings has a single malformed
**first trial** — a session-start artifact where the trial never terminates and
the encoder accumulates position far past the corridor (up to 26508° ≈ 3240 cm
in `DG_S41`, vs a valid ceiling of ~1200° ≈ 147 cm across all datasets). That
one trial holds 13–21 % of all bins; its extreme values wreck the global
standardisation of the position target. Confirmed with the director as a
recording artifact. The existing filters missed it: the immobility filter keeps
the trial's running bins (velocity > `thVel`), and the dPrime filter removed 0
of its bins.

**Decision.** Add a position-based trial filter,
`get_idx_by_out_of_corridor_trials`, to `ibioml.preprocess_data`: any trial
whose **peak position** exceeds `thPosition` degrees (default `1500°` ≈ 183 cm)
is removed in full, union-merged into `indices_to_remove` before
standardisation. Threshold is in raw encoder degrees (independent of the cm
factor). `scripts/preprocess_all.py` passes `thPosition=1500` via
`DEFAULT_CONFIG`.

**Alternatives.**

- The pre-existing duration filter (`get_idx_by_high_trial_duration`, remove
  trials with duration > mean+3σ, currently commented out). Rejected: duration
  conflates the artifact with valid long trials where the animal legitimately
  stays still (bounded position) — e.g. `DG_S38` has a valid 292 s immobility
  trial only ~40 s under its mean+3σ threshold. The threshold is also inflated
  by the outlier itself; it happens to work here by luck of the statistics, not
  by design. The lab abandoned this filter for exactly this reason.
- Bin-level removal of under-represented position ranges. Deferred: the
  artifact's natural unit is the whole trial; bin-level histogram outlier
  removal risks fragmenting trials and dropping legitimately rare far-corridor
  bins. A per-trial position-representation analysis is on the backlog
  independently.
- Editing `data/raw`. Rejected: raw files are source-of-truth and do-not-modify
  (D-013). Removal is a reproducible preprocessing-time operation.

**Why.** Peak position is the discriminator that separates the artifact
(position leaves the corridor) from valid long trials (position bounded);
duration cannot. The threshold sits in a wide empty gap — max valid peak 1203°,
min pathological 2488° — so any value in ~1300–2400° gives identical results.
Verified blast radius: removes exactly one trial from each of
`DG_S38`/`DG_S39`/`DG_S41` and nothing from the other 18 datasets, so only those
three need re-running. Unit-tested
(`external/ibioml/tests/test_out_of_corridor.py`, 5 cases) and confirmed
end-to-end (removes 3861/3195/3789 bins on the three; 0 on a healthy dataset).

Also confirmed as part of this work: the `CM_PER_DEGREE = 0.1222` conversion
factor is correct (verified with the director, 2026-07-14) — updated in
`analysis/loaders.py` and `docs/trial-structure.md`, and removed from the open
questions.

**Follow-up.** The fix currently lives on the fork's working tree. Upstream it
to `mariburginlab-labPrograms/IBioML` via PR once validated by the re-run (per
D-012 test-first, D-002/D-015 submodule flow). Re-run preprocessing + MLP for
the three DG-naive datasets on the GPU host.
