from __future__ import annotations

from gaslight_detector.tasks.schema import Frame, TaskCard


def render_ladder(frames: list[Frame], system_prefix: str | None = None) -> list[dict]:
    rendered = []
    for f in frames:
        prompt = f.prompt.strip()
        if system_prefix:
            prompt = f"{system_prefix.strip()}\n\n{prompt}"
        rendered.append({"frame_id": f.id, "distance": f.distance, "prompt": prompt})
    return rendered
