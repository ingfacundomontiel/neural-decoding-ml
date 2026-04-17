# `specs/` — Project Planning Documents

Living documents that define the project's direction, capture findings, and
record decisions. Always read these first when resuming a session.

## Files

| File | Purpose | When to update |
|---|---|---|
| [`plan.md`](./plan.md) | Goals, non-goals, layout, phased roadmap | When the plan changes |
| [`research.md`](./research.md) | Findings from exploring the codebase and data | As new facts emerge |
| [`decisions.md`](./decisions.md) | Append-only log of architectural decisions | Every non-trivial choice |
| [`dataset-inventory.md`](./dataset-inventory.md) | Catalog of every `.mat` with dedup plan | When datasets change |
| [`fork-ibioml.md`](./fork-ibioml.md) | Runbook for forking `ibioml` and submodule workflow | If the fork strategy shifts |

## Conventions

- **Dates are absolute** (`2026-04-17`, not "yesterday").
- **Decisions are append-only.** Never edit a past decision in place; supersede it
  with a new one and mark the old as `status: superseded by D-###`.
- **Markdown only.** No MkDocs, no Sphinx, no HTML. Renders natively on GitHub.
- **English throughout.**
