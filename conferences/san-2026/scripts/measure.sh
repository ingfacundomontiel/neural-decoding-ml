#!/bin/bash
# Lay the poster out without clipping and report where each column's content ends (cm).
S=${TMPDIR:-/tmp}; P=$(cd "$(dirname "$0")/.." && pwd)
sed -e 's#</style>#body{height:auto!important;overflow:visible!important} main{flex:none!important;align-items:start!important} section{flex:none!important} footer{display:none!important}</style>#' "$P/poster.html" > "$P/_measure.html"
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --hide-scrollbars \
  --allow-file-access-from-files --window-size=3402,7000 --virtual-time-budget=20000 \
  --screenshot="$S/measure.png" "file://$P/_measure.html" >/dev/null 2>&1
rm "$P/_measure.html"
"$P/../../.venv/bin/python" - "$S/measure.png" <<'PY'
import sys, numpy as np
from PIL import Image
a = np.array(Image.open(sys.argv[1]).convert("L")); px = 3402 / 90
for name, (x0, x1) in {"left": (2.5, 44), "right": (46, 87.5)}.items():
    rows = np.where((a[:, int(x0 * px):int(x1 * px)] < 200).any(1))[0]
    b = rows.max() / px
    print(f"{name}: ends at {b:.1f} cm → {'OVER' if b > 122.5 else 'fits'} by {abs(b - 122.5):.1f} cm")
PY
