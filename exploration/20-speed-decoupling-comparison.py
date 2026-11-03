# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Speed decoupling — neural vs speed vs neural + speed position decoders
#
# Per-recording comparison of three position decoders that share the same MLP,
# Optuna search, kept bins and trial folds, and differ only in their input:
#
# - **Neural Act.** — Batch 007 (`results_notail_smooth_w010/`): final pipeline.
# - **Speed** — Batch 011 (`results_behav_speedctx_w010/`): speed window + context.
# - **Neural Act. + Speed** — Batch 012 (`results_neural_speedctx_w010/`): both.
#
# Per recording and decoder: fold-median test R² and fold-median MAE (cm), plus
# group medians. The three studies are asserted to hold identical test targets
# (same bins, same folds), so every comparison is paired.
# Writes `speed-decoupling/comparison.md` (`OUT_PATH` overrides) and the same content
# as a styled page, `speed-decoupling/comparison.html`, in the SAN 2026 poster palette
# with region / cohort colour keys.

# %%
import html
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

from analysis.loaders import CM_PER_DEGREE  # noqa: E402
from analysis.results import (  # noqa: E402
    discover_studies,
    fold_predictions,
    fold_r2_scores,
    load_final_results,
)

DECODERS = {  # header label → results tree
    "Neural Act.": ROOT / "results_notail_smooth_w010",
    "Speed (window)": ROOT / "results_behav_speedctx_w010",
    "Neural Act. + Speed (window)": ROOT / "results_neural_speedctx_w010",
    "Speed (smoothed)": ROOT / "results_behav_speedsmooth_w010",
    "Neural Act. + Speed (smoothed)": ROOT / "results_neural_speedsmooth_w010",
}
# The styled page shows only the symmetric (smoothed-speed) control; the md keeps every decoder.
HTML_DECODERS = ("Neural Act.", "Speed (smoothed)", "Neural Act. + Speed (smoothed)")
COUNT_WORD = {3: "Three", 5: "Five"}
OUT_PATH = Path(os.environ.get("OUT_PATH", ROOT / "speed-decoupling" / "comparison.md"))
OUT_HTML = OUT_PATH.with_suffix(".html")
# SAN 2026 poster palette (conferences/san-2026/scripts/poster_style.py).
REGION_COLOR = {"Pir": "#0B63A5", "CA3": "#8E3B8F", "DG": "#C0581B"}
REGION_NAME = {"Pir": "Piriform cortex", "CA3": "CA3", "DG": "DG"}
FONT_DIR = "../conferences/san-2026/assets/fonts"
# Row tints: the region hue mixed with white — darker for experts, lighter for naive.
TINT = {"expert": 0.24, "naive": 0.10}
REGION_ORDER = ("Pir", "CA3", "DG")
COHORT_ORDER = ("naive", "expert")
GROUPS = {  # summary row label → (regions, cohort or None for both)
    "Piriform · Expert": (("Pir",), "expert"),
    "Piriform · Naive": (("Pir",), "naive"),
    "Hippocampus · Expert": (("CA3", "DG"), "expert"),
    "Hippocampus · Naive": (("CA3", "DG"), "naive"),
    "All 17 recordings": (REGION_ORDER, None),
}

INTRO = """\
# Speed decoupling — position decoding from neural activity, speed, or both

Five position decoders, one per column group, identical in everything but their
input, so every comparison is paired:

- MLP (Optuna, 20 trials; 5 trial-wise outer folds × 1 validation hold-out) 
- Same kept bins (post-context tail removed at 515° ≈ 62.9 cm, Gaussian smoothing σ = 200 ms, no history window) 
- Same trial folds

Experiments:

- **Neural Act.** (Batch 007): smoothed spike counts of the kept neurons in the
  current 200 ms bin, plus the trial's context one-hot (rewarded / unrewarded).
- **Speed (window)** (Batch 011): no neural data. The speed input is a window of five raw
  binned speed values, from two bins before to two bins after the current one
  (t − 400 ms … t + 400 ms), taken on the continuous recording before any bin is
  removed — the same convention as the Gaussian smoothing of the neural input.
  The same context one-hot is appended (7 inputs).
- **Neural Act. + Speed (window)** (Batch 012): the Neural Act. features (without
  their context columns) plus the 7 Speed (window) inputs.
- **Speed (smoothed)** (Batch 013): no neural data. One speed value: the continuous
  raw speed series smoothed with exactly the Gaussian applied to the spike counts
  (σ = 200 ms, same function), sampled at the current bin. Speed and neurons then
  share the same temporal transformation, whereas the window also exposes
  acceleration. The same context one-hot is appended (3 inputs).
- **Neural Act. + Speed (smoothed)** (Batch 014): the Neural Act. features (without
  their context columns) plus the 3 Speed (smoothed) inputs.

Values are medians over the 5 outer test folds: R² and mean absolute error (MAE,
cm). Summary rows are medians of the per-recording values.
"""


# %%
def per_fold_metrics(results_dir: Path, study: str) -> tuple[list[float], list[float], np.ndarray]:
    """Fold R², fold MAE (cm) and the pooled test targets of one study."""
    final = load_final_results(results_dir, study)
    folds = fold_predictions(final)
    mae = [float(np.mean(np.abs((p - t) * CM_PER_DEGREE))) for t, p in folds]
    targets = np.concatenate([t for t, _ in folds])
    return fold_r2_scores(final), mae, targets


def label(info) -> str:
    return f"{info.region}-{info.animal}-{info.cohort.capitalize()}"


def collect() -> list[dict]:
    neural = discover_studies(DECODERS["Neural Act."], "pos")
    neural.sort(key=lambda i: (REGION_ORDER.index(i.region), COHORT_ORDER.index(i.cohort), i.animal))
    rows = []
    for info in neural:
        row = {"label": label(info), "region": info.region, "cohort": info.cohort}
        ref = None
        for name, tree in DECODERS.items():
            r2, mae, targets = per_fold_metrics(tree, info.study)
            if ref is None:
                ref = targets
            elif not np.array_equal(ref, targets):
                raise ValueError(f"{info.study}: test targets differ between decoders — not paired")
            row[name] = (float(np.median(r2)), float(np.median(mae)))
        rows.append(row)
    return rows


def group_rows(rows: list[dict]) -> list[dict]:
    """Summary rows: per-group medians of the per-recording values."""
    out = []
    for group, (regions, cohort) in GROUPS.items():
        sel = [r for r in rows if r["region"] in regions and (cohort is None or r["cohort"] == cohort)]
        out.append({"label": f"{group} (n={len(sel)})", "regions": regions, "cohort": cohort,
                    **{n: (float(np.median([r[n][0] for r in sel])),
                           float(np.median([r[n][1] for r in sel]))) for n in DECODERS}})
    return out


def html_table(rows: list[dict]) -> str:
    head = ['<table>', '<thead>', '<tr><th rowspan="2">Recording</th>']
    head += [f'<th colspan="2">{name}</th>' for name in DECODERS]
    head += ['</tr>', '<tr>' + '<th>R²</th><th>MAE (cm)</th>' * len(DECODERS) + '</tr>', '</thead>', '<tbody>']

    def cells(values, bold=False):
        fmt = (lambda s: f"<b>{s}</b>") if bold else (lambda s: s)
        return "".join(f"<td>{fmt(f'{r2:.3f}')}</td><td>{fmt(f'{mae:.1f}')}</td>" for r2, mae in values)

    body = [f"<tr><td>{r['label']}</td>{cells(r[n] for n in DECODERS)}</tr>" for r in rows]
    for g in group_rows(rows):
        body.append(f"<tr><td><b>{g['label']}</b></td>{cells((g[n] for n in DECODERS), bold=True)}</tr>")
    return "\n".join(head + body + ["</tbody>", "</table>"])


# %% [markdown]
# ## Styled HTML page (same content, poster palette, colour keys)

# %%
def inline_md(text: str) -> str:
    """The two inline markups INTRO uses: **bold** and `code`."""
    text = html.escape(text, quote=False)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", text)


def intro_html(names: tuple[str, ...]) -> str:
    """INTRO as HTML (title, paragraphs, bullet lists), keeping only the experiment bullets
    of the decoders in *names* and restating the decoder count accordingly."""
    blocks, out = INTRO.strip().split("\n\n"), []
    dropped = [n for n in DECODERS if n not in names]
    for block in blocks:
        if block.startswith("# "):
            out.append(f"<h1>{inline_md(block[2:])}</h1>")
        elif block.startswith("- "):
            items = [i for i in re.split(r"\n- ", block[2:])
                     if not any(i.startswith(f"**{n}**") for n in dropped)]
            out.append("<ul>" + "".join(f"<li>{inline_md(' '.join(i.split()))}</li>" for i in items) + "</ul>")
        else:
            text = " ".join(block.split()).replace(f"{COUNT_WORD[len(DECODERS)]} position decoders",
                                                   f"{COUNT_WORD[len(names)]} position decoders")
            out.append(f"<p>{inline_md(text)}</p>")
    return "\n".join(out)


def tint(region: str, cohort: str) -> str:
    """Region colour blended with white by the cohort's TINT weight, as #rrggbb."""
    w, c = TINT[cohort], REGION_COLOR[region].lstrip("#")
    rgb = [round(int(c[i:i + 2], 16) * w + 255 * (1 - w)) for i in (0, 2, 4)]
    return "#" + "".join(f"{v:02X}" for v in rgb)


def row_background(regions: tuple[str, ...], cohort: str | None) -> str:
    """One region → flat tint; several (hippocampus summary) → a left-to-right blend."""
    if cohort is None:
        return "var(--summary)"
    colors = [tint(reg, cohort) for reg in regions]
    return colors[0] if len(colors) == 1 else f"linear-gradient(90deg, {', '.join(colors)})"


def dot(region: str, cohort: str | None) -> str:
    """Poster key: region hue; filled = expert, hollow = naive."""
    color = REGION_COLOR[region]
    fill = color if cohort == "expert" else "#fff"
    return f'<span class="dot" style="border-color:{color};background:{fill}"></span>'


REFERENCE = "Neural Act."   # deltas in the other column groups are relative to this one


def delta_span(change: float, digits: int) -> str:
    """Small change indicator; *change* > 0 means better (higher R², lower MAE)."""
    shown = round(change, digits)
    if shown == 0:
        return '<span class="delta flat">= 0</span>'
    arrow, cls = ("▲", "up") if shown > 0 else ("▼", "down")
    return f'<span class="delta {cls}">{arrow} {abs(shown):.{digits}f}</span>'


def value_cells(row: dict, names: tuple[str, ...]) -> str:
    """R² and MAE cells for the decoders in *names*; non-reference groups carry a delta vs
    REFERENCE. Deltas are taken on the displayed (rounded) values so they match the table."""
    ref_r2, ref_mae = round(row[REFERENCE][0], 3), round(row[REFERENCE][1], 1)
    out = []
    for name in names:
        r2, mae = round(row[name][0], 3), round(row[name][1], 1)
        d_r2 = d_mae = ""
        if name != REFERENCE:
            d_r2, d_mae = delta_span(r2 - ref_r2, 3), delta_span(ref_mae - mae, 1)
        out.append(f"<td>{r2:.3f}{d_r2}</td><td>{mae:.1f}{d_mae}</td>")
    return "".join(out)


def styled_table(rows: list[dict], names: tuple[str, ...]) -> str:
    head = ['<table>', '<thead>', '<tr><th rowspan="2" class="rec">Recording</th>']
    head += [f'<th colspan="2" class="grp">{name}</th>' for name in names]
    head += ['</tr>', '<tr>' + '<th>R²</th><th>MAE (cm)</th>' * len(names) + '</tr>', '</thead>', '<tbody>']
    body, prev = [], None
    for r in rows:
        cls = ' class="region-start"' if prev is not None and r["region"] != prev else ""
        prev = r["region"]
        vals = value_cells(r, names)
        bg = row_background((r["region"],), r["cohort"])
        body.append(f'<tr{cls} style="background:{bg}"><td class="rec">{dot(r["region"], r["cohort"])}'
                    f'{r["label"]}</td>{vals}</tr>')
    for i, g in enumerate(group_rows(rows)):
        cls = "summary first" if i == 0 else "summary"
        dots = "".join(dot(reg, g["cohort"]) for reg in g["regions"]) if g["cohort"] else ""
        vals = value_cells(g, names)
        bg = row_background(g["regions"], g["cohort"])
        body.append(f'<tr class="{cls}" style="background:{bg}"><td class="rec">{dots}{g["label"]}</td>'
                    f'{vals}</tr>')
    return "\n".join(head + body + ["</tbody>", "</table>"])


def group_rules_css(n: int) -> str:
    """Selectors for a vertical rule at every boundary between *n* column groups.
    Header row 1: one th per group after "Recording"; row 2 and body: two cells per group."""
    first = [f"thead tr:first-child th:nth-child({i})" for i in range(3, n + 2)]
    last = [f"thead tr:last-child th:nth-child({2 * i + 1})" for i in range(1, n)]
    body = [f"tbody td:nth-child({2 * i + 2})" for i in range(1, n)]
    return ",\n  ".join(first + last + body) + " { border-left: 2px solid #8C8C8C; }"


def key_html() -> str:
    regions = "".join(f'<span class="k">{dot(reg, "expert")}{REGION_NAME[reg]}</span>' for reg in REGION_ORDER)
    cohorts = (f'<span class="k">{dot("Pir", "naive").replace(REGION_COLOR["Pir"], "#1E1E1E")}naive</span>'
               f'<span class="k">{dot("Pir", "expert").replace(REGION_COLOR["Pir"], "#1E1E1E")}expert</span>')
    deltas = ('<span class="k"><span class="delta up">▲</span>&nbsp;better /&nbsp;<span class="delta down">▼'
              f'</span>&nbsp;worse than {REFERENCE} (higher R², lower MAE)</span>')
    return f'<div class="key">{regions}<span class="sep"></span>{cohorts}<span class="sep"></span>{deltas}</div>'


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Speed decoupling comparison</title>
<style>
  @font-face {{ font-family: "Plex"; src: url("{font}/IBMPlexSans-Regular.ttf"); font-weight: 400; }}
  @font-face {{ font-family: "Plex"; src: url("{font}/IBMPlexSans-SemiBold.ttf"); font-weight: 600; }}
  :root {{ --ink: #1E1E1E; --ink-soft: #5A5A5A; --rule: #D6D6D6; --summary: #F4F4F4; }}
  html, body {{ margin: 0; background: #fff; color: var(--ink); }}
  body {{ font-family: "Plex", "IBM Plex Sans", system-ui, sans-serif; font-size: 16px; line-height: 1.5; }}
  main {{ max-width: 1080px; margin: 0 auto; padding: 32px 16px 48px; }}
  h1 {{ font-size: 26px; font-weight: 600; line-height: 1.25; margin: 0 0 16px; }}
  p, li {{ max-width: 80ch; }}
  ul {{ padding-left: 1.2em; }}
  li {{ margin-bottom: 6px; }}
  code {{ font-size: 0.9em; background: #F2F2F2; padding: 1px 4px; border-radius: 3px; }}
  .key {{ display: flex; flex-wrap: wrap; gap: 8px 20px; align-items: center; margin: 24px 0 12px;
          font-size: 14px; color: var(--ink-soft); }}
  .key .k {{ display: inline-flex; align-items: center; }}
  .key .sep {{ width: 1px; height: 18px; background: var(--rule); }}
  .dot {{ display: inline-block; width: 11px; height: 11px; border: 2.5px solid; border-radius: 50%;
          margin-right: 7px; vertical-align: -1px; box-sizing: border-box; }}
  .table-wrap {{ overflow-x: auto; }}
  .delta {{ font-size: 11px; font-weight: 600; margin-left: 6px; }}
  .key .delta {{ margin-left: 0; font-size: 13px; }}
  .delta.up {{ color: #1B7F3B; }}
  .delta.down {{ color: #B3261E; }}
  .delta.flat {{ color: #8C8C8C; }}
  table {{ border-collapse: collapse; width: auto; font-variant-numeric: tabular-nums; font-size: 15px; }}
  td.rec {{ padding-right: 20px; }}
  th, td {{ padding: 6px 12px; text-align: right; white-space: nowrap; }}
  th {{ font-weight: 600; }}
  th.grp {{ text-align: center; border-bottom: 2px solid var(--ink); white-space: normal; }}
  thead tr:last-child th {{ border-bottom: 1px solid var(--ink); color: var(--ink-soft); font-size: 13px; }}
  th.rec, td.rec {{ text-align: left; }}
  td:nth-child(2n+2), th:nth-child(2n+2) {{ padding-left: 28px; }}
  thead tr:last-child th:nth-child(2n+1) {{ padding-left: 28px; }}
  tbody tr {{ border-bottom: 2px solid #fff; }}
  tbody tr.region-start {{ border-top: 6px solid #fff; }}
  tbody tr.summary {{ font-weight: 600; }}
  tbody tr.summary.first {{ border-top: 2px solid var(--ink); }}
  tbody tr.summary.first td {{ padding-top: 8px; }}
  /* vertical rule between column groups */
  {rules}
</style>
</head>
<body>
<main>
{intro}
{key}
<div class="table-wrap">
{table}
</div>
</main>
</body>
</html>
"""


# %%
rows = collect()
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(INTRO + "\n" + html_table(rows) + "\n")
OUT_HTML.write_text(PAGE.format(font=FONT_DIR, intro=intro_html(HTML_DECODERS), key=key_html(),
                                table=styled_table(rows, HTML_DECODERS),
                                rules=group_rules_css(len(HTML_DECODERS))))
print(f"wrote {OUT_PATH.relative_to(ROOT)} and {OUT_HTML.relative_to(ROOT)} ({len(rows)} recordings)")
