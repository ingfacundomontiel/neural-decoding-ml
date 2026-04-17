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
**Status:** accepted

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
**Status:** accepted

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
