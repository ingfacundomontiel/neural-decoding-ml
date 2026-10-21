"""Schematic diagrams (MLP, trial-wise CV) as inline-SVG snippets for poster.html.

Written as HTML-embeddable <svg> (mm user units) instead of image files so the
text inherits the page's IBM Plex Sans; font sizes are given in printed points.
Run, then `inject()` replaces the <!--MLP--> / <!--CV--> blocks in poster.html.
"""
from __future__ import annotations

import re

from poster_style import FIG_DIR, POSTER

PT = 0.3528  # mm per point
INK, SOFT, LINE = "#1E1E1E", "#5A5A5A", "#BDBDBD"


def text(x, y, s, size=22, anchor="middle", weight=400, fill=INK) -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size * PT:.2f}" text-anchor="{anchor}" '
            f'font-weight="{weight}" fill="{fill}">{s}</text>')


def svg_open(w: float, h: float) -> str:
    return f'<svg viewBox="0 0 {w} {h}" width="{w / 10}cm" height="{h / 10}cm" xmlns="http://www.w3.org/2000/svg">'


def mlp() -> str:
    w, h = 190, 118
    x_in, x_h1, x_h2, x_out, y_out = 48, 84, 112, 142, 57
    ys_in = [22 + i * 9.5 for i in range(6)]
    ys_ctx = [85, 94]
    ys_h = [18 + i * 11 for i in range(8)]
    edges = [(x_in, a, x_h1, b) for a in ys_in + ys_ctx for b in ys_h]
    edges += [(x_h1, a, x_h2, b) for a in ys_h for b in ys_h]
    edges += [(x_h2, a, x_out, y_out) for a in ys_h]
    out = [svg_open(w, h), f'<g stroke="{LINE}" stroke-width="0.25">']
    out += [f'<line x1="{a}" y1="{b}" x2="{c}" y2="{d}"/>' for a, b, c, d in edges]
    out.append("</g>")
    out += [f'<circle cx="{x_in}" cy="{y}" r="3.6" fill="#fff" stroke="{INK}" stroke-width="0.7"/>'
            for y in ys_in]
    out += [f'<rect x="{x_in - 3.6}" y="{y - 3.6}" width="7.2" height="7.2" fill="#fff" '
            f'stroke="{INK}" stroke-width="0.7"/>' for y in ys_ctx]
    out += [f'<circle cx="{x}" cy="{y}" r="3.6" fill="#E6E6E6" stroke="{INK}" stroke-width="0.7"/>'
            for x in (x_h1, x_h2) for y in ys_h]
    out.append(f'<circle cx="{x_out}" cy="{y_out}" r="5" fill="{INK}"/>')
    out += [text(x_in - 8, 45, "N neurons", 24, "end"), text(x_in - 8, 54, "smoothed", 20, "end", fill=SOFT),
            text(x_in - 8, 92, "context", 24, "end"),
            text((x_h1 + x_h2) / 2, 10, "Linear → BatchNorm → ReLU → Dropout", 20, fill=SOFT),
            text((x_h1 + x_h2) / 2, 111, "1–3 hidden layers · 128–512 units", 22),
            text(x_out + 8, 56, "position", 24, "start", 600),
            text(x_out + 8, 65, "(linear, cm)", 20, "start", fill=SOFT), "</svg>"]
    return "".join(out)


def cv() -> str:
    w, h = 190, 118
    x0, x1, n = 36, 186, 10
    wtr = (x1 - x0) / n
    fill = {"train": "#E3E3E3", "val": "#9A9A9A", "test": INK}

    def row(y, kinds, label):
        cells = [f'<rect x="{x0 + i * wtr + 0.4:.1f}" y="{y}" width="{wtr - 0.8:.1f}" height="7" '
                 f'fill="{fill.get(k, "#fff")}" stroke="{INK}" stroke-width="0.4"/>' for i, k in enumerate(kinds)]
        return text(x0 - 4, y + 5.5, label, 22, "end") + "".join(cells)

    folds = (("fold 1", 36, {0, 1}, {7, 8}), ("fold 2", 50, {2, 3}, {0, 9}), ("fold 5", 72, {8, 9}, {4, 5}))
    out = [svg_open(w, h), text((x0 + x1) / 2, 9, "trials, shuffled — each trial stays whole", 22),
           row(13, ["blank"] * n, "trials")]
    for name, y, test, val in folds:
        out.append(row(y, ["test" if i in test else "val" if i in val else "train" for i in range(n)], name))
    out.append(text((x0 + x1) / 2, 67, "⋮", 26))
    for x, k, label in ((x0, "train", "train"), (x0 + 42, "val", "validation"), (x0 + 104, "test", "test")):
        out.append(f'<rect x="{x}" y="90" width="7" height="7" fill="{fill[k]}" stroke="{INK}" stroke-width="0.4"/>')
        out.append(text(x + 10, 96.5, label, 22, "start"))
    out += [text((x0 + x1) / 2, 112, "each test fold → one R² → one dot per box", 20, fill=SOFT), "</svg>"]
    return "".join(out)


def inject() -> None:
    page = POSTER / "poster.html"
    html = page.read_text()
    for tag, snippet in (("MLP", mlp()), ("CV", cv())):
        html = re.sub(rf"<!--{tag}-->.*?<!--/{tag}-->", f"<!--{tag}-->{snippet}<!--/{tag}-->", html,
                      flags=re.S)
    page.write_text(html)


if __name__ == "__main__":
    for name, snippet in (("diagram_mlp", mlp()), ("diagram_cv", cv())):
        (FIG_DIR / f"{name}.svg").write_text(snippet)
    inject()
