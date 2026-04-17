# Neural Decoding ML

Thesis codebase for neural decoding of mouse spatial position from piriform
cortex and hippocampus recordings.

> **Scientific question.** Given neural activity recorded while a mouse navigates
> a virtual-reality corridor (expert/naive piriform cortex or hippocampus CA3/DG),
> can we decode the animal's position — and, if so, how does the decoder's
> performance depend on brain region, number of neurons, temporal binning, and
> trial structure?

This is a working thesis repo. Read [`specs/`](./specs/) before doing anything else.

## Project map

```
specs/          plan, research findings, decisions, dataset inventory
data/raw/       canonical .mat recordings (20 files)
data/processed/ derived pickles (gitignored, reproducible)
external/ibioml/ git submodule → facundomontiel/IBioML fork
exploration/    Jupytext-paired .py notebooks for data exploration
analysis/       reusable Python modules (plotting, stats) supporting exploration
experiments/    reproducible configs — empty for now
docs/           thesis-grade writeups (audit, findings, methods)
tests/          unit tests for analysis/ and ibioml fixes
_legacy/        archive of the previous neural-decoding-ml attempt
```

## Getting started

### Prerequisites

- macOS or Linux
- [Homebrew](https://brew.sh) (for `uv`)
- `git` with submodule support
- `gh` CLI (only needed for contributing fixes back to `ibioml`)

### First-time setup

```bash
# Clone with submodules (important!)
git clone --recurse-submodules <this-repo-url>
cd neural-decoding-ml

# Install uv if you don't have it
brew install uv

# Create the virtualenv and install everything (pinned via uv.lock)
uv sync

# Verify the environment
uv run python -c "import ibioml; print('ok')"
```

### If you already cloned without `--recurse-submodules`

```bash
git submodule update --init --recursive
uv sync
```

### Running a notebook

Exploration notebooks are `.py` files (Jupytext percent format). Open them with
JupyterLab:

```bash
uv run jupyter lab
```

Right-click a `.py` notebook → "Open With" → "Notebook". Jupytext pairs it with
an `.ipynb` on the fly; edits stay in the `.py`.

## Workflow snapshot

- **Reading, not writing?** Start with [`specs/plan.md`](./specs/plan.md).
- **Adding a decision?** Append to [`specs/decisions.md`](./specs/decisions.md).
- **Fixing an `ibioml` bug?** See [`specs/fork-ibioml.md`](./specs/fork-ibioml.md)
  for the branch + PR flow.
- **Adding datasets?** Update [`specs/dataset-inventory.md`](./specs/dataset-inventory.md)
  and [`data/raw/README.md`](./data/raw/README.md) together.

## Status

Phase 0 (bootstrap) complete. Phase 1 (dataset migration + `ibioml` audit) next.
See [`specs/plan.md`](./specs/plan.md) for the roadmap.
