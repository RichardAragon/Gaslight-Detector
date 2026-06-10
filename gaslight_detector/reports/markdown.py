from __future__ import annotations

from typing import Any


def _fmt(v, nd=3):
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)


def render_markdown(result: dict[str, Any]) -> str:
    task = result["task"]
    risk = result["risk_ladder"]
    spike = risk["spike"]
    lines: list[str] = []

    if result.get("synthetic"):
        lines.append("> **SYNTHETIC DEMO — not a real model run.** Responses are hand-authored "
                     "fixtures illustrating the mechanism. No conclusion about any model may be "
                     "drawn from this report.\n")

    if result.get("exploratory_only"):
        lines.append("> **EXPLORATORY ONLY — no topic-selective conclusion.** "
                     + result.get("exploratory_reason", "") + "\n")

    lines.append(f"# Gaslight Detector Report: {task['title']}\n")
    lines.append(f"**Task ID:** `{task['id']}`  ")
    lines.append(f"**Provider / Model:** `{result['provider']}` / `{result['model']}`  ")
    lines.append(f"**Risk axis:** `{task['risk_axis']}`  ")
    lines.append(f"**Schema:** `{result['schema_version']}`  •  **Tool:** `{result['gaslight_detector_version']}`\n")

    # Scorer provenance (point 6: every scorer + version shown separately, never collapsed).
    lines.append("## Scorers\n")
    lines.append("| Scorer | Version | Learned model? |")
    lines.append("|---|---|---|")
    for sc in result.get("scorers", []):
        lines.append(f"| `{sc['name']}` | `{sc['version']}` | {'yes' if sc['learned'] else 'no'} |")
    if not any(sc["learned"] for sc in result.get("scorers", [])):
        lines.append("\n> ⚠ No learned scorer was active — substance is from lexical/hashing "
                     "approximations only. Treat substance numbers as a transparency floor, not a "
                     "semantic ground truth. Install `gaslight-detector[embedding]` for a learned "
                     "embedder.")
    lines.append("")

    lines.append("## Result\n")
    ci = spike.get("max_drop_ci")
    ci_txt = f" (90% CI [{_fmt(ci[0])}, {_fmt(ci[1])}])" if ci else ""
    lines.append(f"- **Gaslight Spike:** {'YES' if spike['flag'] else 'no'}  ")
    lines.append(f"- **Strength:** `{spike['strength']}`  ")
    lines.append(f"- **Max drop (composite MPS):** `{_fmt(spike['max_drop'])}`{ci_txt}  ")
    lines.append(f"- **Robust ratio:** `{_fmt(spike['robust_ratio'], 2)}`  ")
    ti = spike["transition_index"]
    if ti is not None and ti + 1 < len(risk["frame_ids"]):
        lines.append(f"- **Largest discontinuity:** `{risk['frame_ids'][ti]}` → `{risk['frame_ids'][ti+1]}`\n")

    diff = result.get("differential") or {}
    if diff:
        lines.append("## Control-normalised differential\n")
        lines.append(f"- Risk-ladder max drop: `{_fmt(diff['risk_max_drop'])}`  ")
        lines.append(f"- Harmless-control max drop: `{_fmt(diff['harmless_max_drop'])}`  ")
        lines.append(f"- **Differential:** `{_fmt(diff['differential_max_drop'])}`  ")
        lines.append(f"- _{diff['interpretation']}_\n")
    else:
        lines.append("> No harmless control ladder was scored. A standalone spike is **not** a "
                     "publishable result; add a harmless control and re-run.\n")

    # Per-scorer substance columns, kept separate.
    sub_by = risk.get("scorer_substance_by_frame", {})
    sub_names = list(sub_by.keys())
    lines.append("## Risk ladder — per-frame scores\n")
    header = "| Frame | Dist | MPS | Composite subst. | " + " | ".join(f"{n} subst." for n in sub_names)
    header += " | Sem. shell | Hedging | Fulfil. |"
    lines.append(header)
    lines.append("|---|---:|---:|---:|" + "---:|" * len(sub_names) + "---:|---:|---:|")
    fulfil = risk["fulfillment_by_frame"]
    for i, fid in enumerate(risk["frame_ids"]):
        ful = _fmt(fulfil[i]) if fulfil[i] is not None else "—"
        row = (f"| `{fid}` | {_fmt(risk['distances'][i],2)} | {_fmt(risk['mps_by_frame'][i])} | "
               f"{_fmt(risk['composite_substance_by_frame'][i])} | ")
        row += " | ".join(_fmt(sub_by[n][i]) for n in sub_names)
        row += (f" | {_fmt(risk['semantic_shell_by_frame'][i])} | "
                f"{_fmt(risk['hedging_by_frame'][i])} | {ful} |")
        lines.append(row)
    lines.append("")

    lines.append("## Transition geometry\n")
    has_cont = any("semantic_continuity" in t for t in risk["transitions"])
    h = "| Transition | Struct. dist | Shell dist | Ghost ratio |" + (" Sem. continuity |" if has_cont else "")
    lines.append(h)
    lines.append("|---|---:|---:|---:|" + ("---:|" if has_cont else ""))
    for t in risk["transitions"]:
        i = t["transition_index"]
        row = (f"| `{risk['frame_ids'][i]}` → `{risk['frame_ids'][i+1]}` | "
               f"{_fmt(t['structural_distance'])} | {_fmt(t['semantic_shell_distance'])} | "
               f"{_fmt(t['ghost_ratio'],2)} |")
        if has_cont:
            row += f" {_fmt(t.get('semantic_continuity'))} |"
        lines.append(row)
    lines.append("")

    lines.append("## Interpretation discipline\n")
    lines.append(result["disclaimer"] + "\n")
    lines.append("Possible causes of a spike include safety-policy shaping, hidden system "
                 "instructions, prompt sensitivity, training/RLHF artifacts, provider-side routing, "
                 "topic-specific refusal policy, genuine model uncertainty, or intentional "
                 "degradation. This tool does not distinguish among them.\n")
    return "\n".join(lines)
