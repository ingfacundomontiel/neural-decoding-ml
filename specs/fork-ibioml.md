# Forking `ibioml` — Step-by-Step Runbook

For reference during Phase 0 bootstrap. Commands assume macOS + `zsh` + `gh` CLI
authenticated as `facundomontiel`. Run these one at a time and check output.

## Upstream facts

- Upstream repo: `https://github.com/mariburginlab-labPrograms/IBioML.git`
- Upstream org: `mariburginlab-labPrograms` (note: typo — "mari" not "marin")
- Fork target: `https://github.com/facundomontiel/IBioML.git`
- Local checkout of upstream exists at: `/Users/facundo/Development/Academia/IBioBA/Libreria/IBioML`

## Step 1 — Create the fork on GitHub

```bash
gh repo fork mariburginlab-labPrograms/IBioML \
  --clone=false \
  --default-branch-only
```

- `--clone=false` — don't clone locally yet, we do that via submodule
- `--default-branch-only` — only fork main, no stale feature branches

Verify:

```bash
gh repo view facundomontiel/IBioML --json name,parent,defaultBranchRef
```

## Step 2 — Add the fork as a submodule inside the project

From inside `/Users/facundo/Development/Academia/IBioBA/neural-decoding-ml/`:

```bash
git submodule add git@github.com:facundomontiel/IBioML.git external/ibioml
```

Then, inside the submodule, add upstream as a second remote:

```bash
git -C external/ibioml remote add upstream https://github.com/mariburginlab-labPrograms/IBioML.git
git -C external/ibioml fetch upstream
```

## Step 3 — Pin to the upstream commit you're auditing from

```bash
git -C external/ibioml checkout 1362805
git -C external/ibioml switch -c thesis/audit-baseline
```

Commit the submodule pointer in the parent repo:

```bash
git add .gitmodules external/ibioml
git commit -m "chore: add ibioml fork as submodule pinned to audit baseline"
```

## Step 4 — Workflow for fixes

For every fix found during audit:

```bash
# Inside the submodule
git -C external/ibioml switch -c fix/index-type-preprocess-data
# ... edit files ...
git -C external/ibioml commit -am "fix: cast trial-removal indices to int in preprocess_data"
git -C external/ibioml push -u origin fix/index-type-preprocess-data

# Open PR to upstream
gh pr create --repo mariburginlab-labPrograms/IBioML \
  --head facundomontiel:fix/index-type-preprocess-data \
  --base main \
  --title "Fix: cast trial-removal indices to int in preprocess_data" \
  --body "Fixes 'arrays used as indices must be of integer type' error ..."
```

Back in the parent repo, update the submodule pointer and commit:

```bash
git -C external/ibioml checkout thesis/audit-baseline
git -C external/ibioml merge fix/index-type-preprocess-data
git add external/ibioml
git commit -m "chore: bump ibioml submodule to include index-type fix"
```

## Step 5 — Install the submodule into the uv environment

Inside `pyproject.toml`:

```toml
[tool.uv.sources]
ibioml = { path = "external/ibioml", editable = true }

[project]
dependencies = [
  "ibioml",
  # ...
]
```

Then:

```bash
uv sync
```

This gives an editable install so local changes to `external/ibioml` are picked
up without reinstalling.

## Notes

- **Never edit `ibioml` via a pip-installed copy.** Always edit the submodule
  source; the editable install reflects changes immediately.
- **Keep the `thesis/audit-baseline` branch.** It's the single branch on the fork
  that the thesis results track. Merge fix branches into it after they're
  reviewed upstream (or after they're confirmed working if upstream is slow).
- **Cloning the parent repo elsewhere** requires `git clone --recurse-submodules`
  or a follow-up `git submodule update --init --recursive`.
