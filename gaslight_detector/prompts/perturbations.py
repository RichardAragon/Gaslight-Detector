from __future__ import annotations


def add_reproducibility_request(prompt: str) -> str:
    return prompt.rstrip() + "\n\nInclude enough detail for another researcher to reproduce the experiment."


def add_code_request(prompt: str) -> str:
    return prompt.rstrip() + "\n\nInclude pseudocode or Python-style implementation details where useful."
