# Copyright 2025-2026 FIZ Karlsruhe - Leibniz-Institut für Informationsinfrastruktur GmbH
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Inline-SVG charts for report.py - no JavaScript chart library, no network.

Colour encodes the *model* everywhere (slot 1, slot 2, ... in run order); the
mode is a secondary encoding (exact = hollow marker / dashed line, fuzzy =
filled / solid). Every mark carries a data-tip attribute that the report's
tooltip script shows on hover.
"""

import html
from typing import Dict, List, Optional, Sequence, Tuple

# Categorical slots in fixed order; the CSS defines light and dark values.
SERIES = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)"]

CHART_CSS = """
:root { --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --grid:#e1e0d9; --axis:#c3c2b7; --surface:#fcfcfb; }
@media (prefers-color-scheme: dark) { :root { --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
  --grid:#2c2c2a; --axis:#383835; --surface:#1a1a19; } }
.charts { display:grid; grid-template-columns:repeat(auto-fit, minmax(min(100%, 520px), 1fr)); gap:16px; align-items:start; }
.card { border:1px solid var(--line); border-radius:8px; padding:14px 16px 10px; background:var(--surface); min-width:0; }
.card h3 { font-size:14px; margin:0 0 2px; } .card .sub { color:var(--muted); font-size:12.5px; margin:0 0 8px; }
.card svg { width:100%; height:auto; display:block; overflow:visible; }
.card svg text { font:12px system-ui,-apple-system,"Segoe UI",sans-serif; fill:var(--muted); }
.card svg text.ink { fill:var(--fg); } .card svg text.in { fill:#fff; } .card svg text.in.dark { fill:#0b0b0b; }
.card svg .grid { stroke:var(--grid); stroke-width:1; } .card svg .base { stroke:var(--axis); stroke-width:1; }
.card svg [data-tip] { cursor:default; }
.keys { display:flex; flex-wrap:wrap; gap:4px 14px; font-size:12px; color:var(--muted); margin:0 0 6px; }
.keys span { display:inline-flex; align-items:center; gap:6px; }
.keys svg { width:22px; height:10px; display:inline; }
.tip { position:fixed; z-index:10; pointer-events:none; display:none; white-space:pre; background:var(--fg); color:var(--bg);
  font-size:12px; line-height:1.4; padding:6px 8px; border-radius:6px; box-shadow:0 2px 8px rgba(0,0,0,.2); }
"""

CHART_JS = """
(() => {
  const tip = document.createElement('div'); tip.className = 'tip'; document.body.append(tip);
  document.addEventListener('mouseover', e => {
    const t = e.target.closest('[data-tip]');
    if (!t) { tip.style.display = 'none'; return; }
    tip.textContent = t.dataset.tip; tip.style.display = 'block';
  });
  document.addEventListener('mousemove', e => {
    if (tip.style.display !== 'block') return;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    let x = e.clientX + 14, y = e.clientY + 14;
    if (x + w > innerWidth - 8) x = e.clientX - w - 14;
    if (y + h > innerHeight - 8) y = e.clientY - h - 14;
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  });
})();
"""


def _e(value) -> str:
    return html.escape(str(value), quote=True)


def pct(value: Optional[float]) -> str:
    return "–" if value is None else f"{value * 100:.0f}%"


def card(title: str, subtitle: str, body: str, keys: str = "") -> str:
    return f'<div class="card"><h3>{_e(title)}</h3><p class="sub">{_e(subtitle)}</p>{keys}{body}</div>'


def keys_html(items: Sequence[Tuple[str, str]]) -> str:
    """Legend: (label, swatch-svg-inner) pairs."""
    return '<div class="keys">' + "".join(
        f'<span><svg viewBox="0 0 22 10" aria-hidden="true">{swatch}</svg>{_e(label)}</span>' for label, swatch in items
    ) + "</div>"


def swatch_bar(color: str) -> str:
    return f'<rect x="2" y="1" width="18" height="8" rx="2" fill="{color}"/>'


def swatch_dot(color: str, hollow: bool) -> str:
    fill = "var(--surface)" if hollow else color
    return f'<circle cx="11" cy="5" r="4" fill="{fill}" stroke="{color}" stroke-width="2"/>'


def swatch_line(color: str, dashed: bool) -> str:
    dash = ' stroke-dasharray="4 3"' if dashed else ""
    return f'<line x1="1" y1="5" x2="21" y2="5" stroke="{color}" stroke-width="2" stroke-linecap="round"{dash}/>'


def _hbar(x: float, y: float, w: float, h: float, color: str, tip: str, r: float = 4) -> str:
    """Horizontal bar, square at the baseline, rounded data end."""
    if w <= 0:
        return ""
    r = min(r, w, h / 2)
    path = f"M{x:.1f},{y:.1f}h{w - r:.1f}a{r},{r} 0 0 1 {r},{r}v{h - 2 * r:.1f}a{r},{r} 0 0 1 {-r},{r}h{-(w - r):.1f}z"
    return f'<path d="{path}" fill="{color}" data-tip="{_e(tip)}"/>'


def _col(x: float, base: float, w: float, h: float, color: str, r: float = 4) -> str:
    """Column growing up from `base`, rounded top."""
    if h <= 0:
        return ""
    r = min(r, h, w / 2)
    y = base - h
    path = f"M{x:.1f},{base:.1f}v{-(h - r):.1f}a{r},{r} 0 0 1 {r},{-r}h{w - 2 * r:.1f}a{r},{r} 0 0 1 {r},{r}v{h - r:.1f}z"
    return f'<path d="{path}" fill="{color}"/>'


def _nice_step(value: float) -> int:
    """Tick step so that 4 steps cover `value` with round numbers."""
    for step in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000):
        if value <= step * 4:
            return step
    return int(value / 4) + 1


# -------------------------------
# Charts
# -------------------------------
def dumbbell(rows: List[Dict], title: str, subtitle: str) -> str:
    """One row per model: hollow dot = exact, filled dot = fuzzy, on a 0-100% axis.

    rows: {label, color, exact, fuzzy, exact_tip, fuzzy_tip}
    """
    left, right, row_h, top = 150, 600, 34, 8
    width = right + 40
    height = top + row_h * len(rows) + 24
    scale = lambda v: left + (right - left) * v  # noqa: E731
    out = []
    for t in (0, 0.25, 0.5, 0.75, 1):
        x = scale(t)
        out.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height - 22}"/>')
        out.append(f'<text x="{x:.1f}" y="{height - 6}" text-anchor="middle">{t * 100:.0f}%</text>')
    for i, row in enumerate(rows):
        y = top + row_h * i + row_h / 2
        out.append(f'<text class="ink" x="{left - 12}" y="{y + 4:.1f}" text-anchor="end">{_e(row["label"])}</text>')
        ex, fz = row["exact"], row["fuzzy"]
        if ex is None or fz is None:
            continue
        x1, x2 = scale(ex), scale(fz)
        out.append(f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y:.1f}" stroke="{row["color"]}" stroke-width="2" stroke-opacity=".45"/>')
        for x, hollow, tip in ((x1, True, row["exact_tip"]), (x2, False, row["fuzzy_tip"])):
            fill = "var(--surface)" if hollow else row["color"]
            out.append(f'<g data-tip="{_e(tip)}"><circle cx="{x:.1f}" cy="{y:.1f}" r="12" fill="transparent"/>'
                       f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="var(--surface)"/>'
                       f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{fill}" stroke="{row["color"]}" stroke-width="2"/></g>')
        lo, hi = sorted((x1, x2))
        if hi - lo < 1:
            out.append(f'<text class="ink" x="{hi + 12:.1f}" y="{y + 4:.1f}">{pct(fz)}</text>')
        else:
            out.append(f'<text x="{lo - 10:.1f}" y="{y + 4:.1f}" text-anchor="end">{pct(min(ex, fz))}</text>')
            out.append(f'<text class="ink" x="{hi + 10:.1f}" y="{y + 4:.1f}">{pct(max(ex, fz))}</text>')
    svg = f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_e(title)}">{"".join(out)}</svg>'
    colors = list(dict.fromkeys(r["color"] for r in rows))
    keys = keys_html([("exact (distance 0)", swatch_dot("var(--muted)", True)), ("fuzzy (default threshold)", swatch_dot("var(--muted)", False))]
                     + [(r["label"], swatch_bar(r["color"])) for r in rows if r["color"] in colors])
    return card(title, subtitle, svg, keys)


def histograms(panels: List[Dict], x_values: List[int], title: str, subtitle: str, x_label: str, y_label: str) -> str:
    """Small multiples of column histograms on a shared y scale.

    panels: {label, color, counts: {x: n}, marker: optional x (drawn as median line), tip_unit}
    """
    panel_w, panel_h, gap = 300, 150, 24
    left, bottom, top = 34, 34, 22
    cols = 2 if len(panels) > 1 else 1
    rows = (len(panels) + cols - 1) // cols
    width = cols * panel_w + (cols - 1) * gap
    height = rows * (panel_h + top + bottom) + (rows - 1) * 8
    step = _nice_step(max([max(p["counts"].values(), default=0) for p in panels] + [1]))
    y_max = step * 4
    out = []
    for i, panel in enumerate(panels):
        ox = (i % cols) * (panel_w + gap)
        oy = (i // cols) * (panel_h + top + bottom + 8)
        base = oy + top + panel_h
        plot_w = panel_w - left - 6
        band = plot_w / len(x_values)
        bar_w = min(24, band - 4)
        out.append(f'<text class="ink" x="{ox}" y="{oy + 12}" style="font-weight:600">{_e(panel["label"])}</text>')
        for t in range(0, y_max + 1, step):
            y = base - panel_h * t / y_max
            out.append(f'<line class="grid" x1="{ox + left}" y1="{y:.1f}" x2="{ox + panel_w - 6}" y2="{y:.1f}"/>')
            out.append(f'<text x="{ox + left - 6}" y="{y + 4:.1f}" text-anchor="end">{t}</text>')
        for j, xv in enumerate(x_values):
            cx = ox + left + band * j + band / 2
            n = panel["counts"].get(xv, 0)
            h = panel_h * n / y_max
            out.append(f'<g data-tip="{_e(f"{panel["label"]}{chr(10)}{x_label} {xv}: {n} {panel["tip_unit"]}")}">'
                       f'<rect x="{cx - band / 2:.1f}" y="{base - panel_h:.1f}" width="{band:.1f}" height="{panel_h}" fill="transparent"/>'
                       f'{_col(cx - bar_w / 2, base, bar_w, h, panel["color"])}</g>')
            out.append(f'<text x="{cx:.1f}" y="{base + 14}" text-anchor="middle">{xv}</text>')
        out.append(f'<line class="base" x1="{ox + left}" y1="{base}" x2="{ox + panel_w - 6}" y2="{base}"/>')
        if panel.get("marker") is not None:
            m = panel["marker"]  # x_values are consecutive integers, so a fractional median sits between columns
            mx = ox + left + band * (m - x_values[0]) + band / 2
            out.append(f'<line x1="{mx:.1f}" y1="{base - panel_h}" x2="{mx:.1f}" y2="{base}" stroke="var(--fg)" stroke-width="1" stroke-opacity=".55"/>')
            flip = mx > ox + panel_w - 70  # keep the label inside the panel
            out.append(f'<text class="ink" x="{mx + (-4 if flip else 4):.1f}" y="{base - panel_h + 10}"'
                       f'{" text-anchor=\"end\"" if flip else ""}>median {m:g}</text>')
        out.append(f'<text x="{ox + left + plot_w / 2:.1f}" y="{base + 30}" text-anchor="middle">{_e(x_label)}</text>')
    svg = f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_e(title)}">{"".join(out)}</svg>'
    return card(title, f"{subtitle} Y axis: {y_label}.", svg)


def lines(series: List[Dict], x_values: List[int], title: str, subtitle: str, x_label: str) -> str:
    """Share (0-1) per x for several series. series: {label, color, dashed, points: {x: (share, n_linked, n_total)}}"""
    left, right, top, bottom = 40, 600, 10, 36
    width, height = right + 20, 250
    plot_h = height - top - bottom
    sx = lambda x: left + (right - left) * (x - x_values[0]) / max(1, x_values[-1] - x_values[0])  # noqa: E731
    sy = lambda v: top + plot_h * (1 - v)  # noqa: E731
    out = []
    for t in (0, 0.25, 0.5, 0.75, 1):
        out.append(f'<line class="{"base" if t == 0 else "grid"}" x1="{left}" y1="{sy(t):.1f}" x2="{right}" y2="{sy(t):.1f}"/>')
        out.append(f'<text x="{left - 6}" y="{sy(t) + 4:.1f}" text-anchor="end">{t * 100:.0f}%</text>')
    for x in x_values:
        out.append(f'<text x="{sx(x):.1f}" y="{height - bottom + 16}" text-anchor="middle">{x}</text>')
    out.append(f'<text x="{(left + right) / 2:.1f}" y="{height - 4}" text-anchor="middle">{_e(x_label)}</text>')
    for s in series:
        pts = [(x, s["points"][x]) for x in x_values if x in s["points"]]
        if not pts:
            continue
        d = " ".join(f'{"M" if i == 0 else "L"}{sx(x):.1f},{sy(v[0]):.1f}' for i, (x, v) in enumerate(pts))
        dash = ' stroke-dasharray="5 4"' if s["dashed"] else ""
        out.append(f'<path d="{d}" fill="none" stroke="{s["color"]}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"{dash}/>')
    for s in series:
        for x, (share, linked, total) in ((x, s["points"][x]) for x in x_values if x in s["points"]):
            fill = "var(--surface)" if s["dashed"] else s["color"]
            tip = f'{s["label"]}\nrank {x}: {linked} of {total} linked ({pct(share)})'
            out.append(f'<g data-tip="{_e(tip)}"><circle cx="{sx(x):.1f}" cy="{sy(share):.1f}" r="10" fill="transparent"/>'
                       f'<circle cx="{sx(x):.1f}" cy="{sy(share):.1f}" r="6" fill="var(--surface)"/>'
                       f'<circle cx="{sx(x):.1f}" cy="{sy(share):.1f}" r="4" fill="{fill}" stroke="{s["color"]}" stroke-width="2"/></g>')
    svg = f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_e(title)}">{"".join(out)}</svg>'
    keys = keys_html([(s["label"], swatch_line(s["color"], s["dashed"])) for s in series])
    return card(title, subtitle, svg, keys)


def grouped_hbars(groups: List[Tuple[str, List[Dict]]], title: str, subtitle: str) -> str:
    """groups: [(group label, [{label, color, value (0-1), tip}])] on a 0-100% axis."""
    left, right, bar_h, gap, group_gap, top = 150, 590, 16, 2, 16, 4
    n_bars = sum(len(bars) for _, bars in groups)
    height = top + n_bars * (bar_h + gap) + group_gap * len(groups) + 22
    width = right + 50
    scale = lambda v: (right - left) * v  # noqa: E731
    out = []
    plot_bottom = height - 22
    for t in (0, 0.25, 0.5, 0.75, 1):
        x = left + scale(t)
        out.append(f'<line class="{"base" if t == 0 else "grid"}" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{plot_bottom}"/>')
        out.append(f'<text x="{x:.1f}" y="{height - 6}" text-anchor="middle">{t * 100:.0f}%</text>')
    y = top + group_gap / 2
    legend = {}
    for label, bars in groups:
        group_h = len(bars) * (bar_h + gap) - gap
        out.append(f'<text class="ink" x="{left - 12}" y="{y + group_h / 2 + 4:.1f}" text-anchor="end">{_e(label)}</text>')
        for bar in bars:
            legend[bar["label"]] = bar["color"]
            value = bar["value"] or 0
            out.append(f'<g data-tip="{_e(bar["tip"])}"><rect x="{left}" y="{y:.1f}" width="{right - left}" height="{bar_h}" fill="transparent"/>'
                       f'{_hbar(left, y, scale(value), bar_h, bar["color"], bar["tip"])}</g>')
            out.append(f'<text x="{left + scale(value) + 6:.1f}" y="{y + bar_h - 4:.1f}">{pct(bar["value"])}</text>')
            y += bar_h + gap
        y += group_gap
    svg = f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_e(title)}">{"".join(out)}</svg>'
    keys = keys_html([(label, swatch_bar(color)) for label, color in legend.items()])
    return card(title, subtitle, svg, keys)


def stacked_hbars(rows: List[Tuple[str, Dict[str, int]]], categories: List[Tuple[str, str, bool]], title: str, subtitle: str) -> str:
    """rows: [(label, {category: count})]; categories: [(name, color, dark_label_text)]. Bars are 100% stacks."""
    left, right, bar_h, row_h, top = 150, 600, 22, 32, 4
    width, height = right + 20, top + row_h * len(rows) + 4
    out = []
    for i, (label, counts) in enumerate(rows):
        y = top + row_h * i + (row_h - bar_h) / 2
        total = sum(counts.values())
        out.append(f'<text class="ink" x="{left - 12}" y="{y + bar_h / 2 + 4:.1f}" text-anchor="end">{_e(label)}</text>')
        x = left
        present = [(name, color, dark) for name, color, dark in categories if counts.get(name)]
        for j, (name, color, dark) in enumerate(present):
            n = counts[name]
            w = (right - left) * n / total
            seg_w = w - (2 if j < len(present) - 1 else 0)  # 2px surface gap between segments
            tip = f"{label}\n{name}: {n} of {total} linked ({pct(n / total)})"
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(seg_w, 0):.1f}" height="{bar_h}" rx="4" fill="{color}" data-tip="{_e(tip)}"/>')
            text = f"{name} {n}"
            if seg_w > len(text) * 6.5 + 12:
                out.append(f'<text class="in{" dark" if dark else ""}" x="{x + 8:.1f}" y="{y + bar_h / 2 + 4:.1f}" pointer-events="none">{_e(text)}</text>')
            x += w
    svg = f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_e(title)}">{"".join(out)}</svg>'
    keys = keys_html([(name, swatch_bar(color)) for name, color, _ in categories])
    return card(title, subtitle, svg, keys)
