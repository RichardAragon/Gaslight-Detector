from __future__ import annotations

import html
from typing import Any


def render_html(result: dict[str, Any], plots: list[str] | None = None) -> str:
    task = result["task"]
    risk = result["risk_ladder"]
    spike = risk["spike"]

    def esc(x):
        return html.escape(str(x))

    banners = ""
    if result.get("synthetic"):
        banners += ('<div class="banner synthetic">SYNTHETIC DEMO — not a real model run. '
                    'Hand-authored fixtures illustrating the mechanism. No conclusion about any '
                    'model may be drawn.</div>')
    if result.get("exploratory_only"):
        banners += ('<div class="banner exploratory">EXPLORATORY ONLY — no topic-selective '
                    f'conclusion. {esc(result.get("exploratory_reason",""))}</div>')

    scorer_rows = "".join(
        f"<tr><td>{esc(s['name'])}</td><td>{esc(s['version'])}</td>"
        f"<td>{'yes' if s['learned'] else 'no'}</td></tr>" for s in result.get("scorers", []))
    learned_warn = ""
    if not any(s["learned"] for s in result.get("scorers", [])):
        learned_warn = ('<p class="warn">No learned scorer active — substance is a lexical/hashing '
                        'approximation, not semantic ground truth.</p>')

    sub_by = risk.get("scorer_substance_by_frame", {})
    sub_names = list(sub_by.keys())
    head = ("<th>Frame</th><th>Dist</th><th>MPS</th><th>Composite</th>"
            + "".join(f"<th>{esc(n)}</th>" for n in sub_names)
            + "<th>Shell</th><th>Hedging</th><th>Fulfil.</th>")
    rows = ""
    fulfil = risk["fulfillment_by_frame"]
    for i, fid in enumerate(risk["frame_ids"]):
        ful = f"{fulfil[i]:.3f}" if fulfil[i] is not None else "—"
        cells = "".join(f"<td>{sub_by[n][i]:.3f}</td>" for n in sub_names)
        rows += (f"<tr><td>{esc(fid)}</td><td>{risk['distances'][i]:.2f}</td>"
                 f"<td>{risk['mps_by_frame'][i]:.3f}</td>"
                 f"<td>{risk['composite_substance_by_frame'][i]:.3f}</td>{cells}"
                 f"<td>{risk['semantic_shell_by_frame'][i]:.3f}</td>"
                 f"<td>{risk['hedging_by_frame'][i]:.3f}</td><td>{ful}</td></tr>")

    diff = result.get("differential") or {}
    diff_html = ""
    if diff:
        diff_html = (f'<div class="card"><h2>Control-normalised differential</h2>'
                     f'<p>Risk max drop <b>{diff["risk_max_drop"]:.3f}</b> − harmless '
                     f'<b>{diff["harmless_max_drop"]:.3f}</b> = <b>{diff["differential_max_drop"]:.3f}</b></p>'
                     f'<p><em>{esc(diff["interpretation"])}</em></p></div>')

    plot_imgs = "".join(f'<img src="plots/{esc(p)}" alt="{esc(p)}"/>' for p in (plots or []))
    verdict = "YES" if spike["flag"] else "no"
    color = "#b00020" if spike["flag"] else "#1b7a3d"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Gaslight Detector — {esc(task['title'])}</title>
<style>
  body {{ font-family:-apple-system,system-ui,sans-serif; margin:0; background:#f6f7f9; color:#1a1a1a; }}
  .wrap {{ max-width:980px; margin:0 auto; padding:24px; }}
  .banner {{ padding:12px 16px; border-radius:8px; font-weight:600; margin-bottom:12px; }}
  .synthetic {{ background:#ffe9a8; border:1px solid #d8b53a; }}
  .exploratory {{ background:#dbe9ff; border:1px solid #6a9ce0; }}
  .card {{ background:#fff; border:1px solid #e3e6ea; border-radius:12px; padding:20px; margin-bottom:16px; }}
  .verdict {{ font-size:28px; font-weight:700; color:{color}; }}
  table {{ border-collapse:collapse; width:100%; font-size:13px; }}
  th,td {{ border-bottom:1px solid #eee; padding:7px 9px; text-align:right; }}
  th:first-child, td:first-child {{ text-align:left; }}
  code {{ background:#f0f0f3; padding:2px 5px; border-radius:4px; }}
  img {{ max-width:100%; border:1px solid #e3e6ea; border-radius:8px; margin-top:12px; }}
  .muted {{ color:#666; font-size:13px; }} .warn {{ color:#9a6a00; font-weight:600; }}
</style></head>
<body><div class="wrap">
{banners}
<div class="card">
  <h1>Gaslight Detector: {esc(task['title'])}</h1>
  <p class="muted"><code>{esc(task['id'])}</code> • {esc(result['provider'])}/{esc(result['model'])}
   • risk axis <code>{esc(task['risk_axis'])}</code> • schema {esc(result['schema_version'])}</p>
  <p class="verdict">Gaslight Spike: {verdict} <span class="muted">({esc(spike['strength'])})</span></p>
  <p>Max drop <b>{spike['max_drop']:.3f}</b>, robust ratio <b>{spike['robust_ratio']:.2f}</b>. {esc(spike['message'])}</p>
</div>
<div class="card"><h2>Scorers</h2>
<table><thead><tr><th>Scorer</th><th>Version</th><th>Learned?</th></tr></thead><tbody>{scorer_rows}</tbody></table>
{learned_warn}</div>
{diff_html}
<div class="card"><h2>Risk ladder</h2>
<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>
<div class="card"><h2>Plots</h2>{plot_imgs or '<p class="muted">No plots (matplotlib not installed).</p>'}</div>
<div class="card"><h2>Interpretation discipline</h2><p>{esc(result['disclaimer'])}</p></div>
</div></body></html>"""
