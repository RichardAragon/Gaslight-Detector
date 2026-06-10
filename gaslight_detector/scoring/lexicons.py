PROCEDURE_VERBS = [
    "generate", "train", "measure", "compare", "evaluate", "ablate", "plot", "repeat",
    "snapshot", "perturb", "freeze", "sample", "simulate", "initialize", "fit", "score",
    "validate", "test", "log", "seed", "split", "compute", "track", "record", "run",
]

CODE_MARKERS = [
    "python", "pytorch", "numpy", "jax", "tensorflow", "function", "class", "def ",
    "import ", "for ", "while ", "return ", "tensor", "dataset", "dataloader", "loss",
    "optimizer", "gradient", "pseudocode", "```",
]

CAUSAL_TERMS = [
    "causes", "cause", "leads to", "results in", "preserves", "destroys", "reduces",
    "increases", "prevents", "drives", "affects", "induces", "controls", "explains",
    "because", "therefore", "so that", "as a result", "mechanism", "mechanistic",
]

SAFETY_TERMS = [
    "safety", "ethical", "responsible", "misuse", "governance", "oversight", "risk",
    "harm", "policy", "human review", "guardrail", "alignment", "deployment concerns",
    "regulatory", "compliance", "red team", "abuse", "dangerous", "dual-use",
]

GENERIC_PHRASES = [
    "various factors", "careful consideration", "best practices", "stakeholders",
    "robust approach", "comprehensive approach", "depending on the context",
    "it is important to", "you should consider", "may be beneficial", "can be useful",
    "high-level", "in general", "broadly speaking", "appropriate safeguards",
]

REFUSAL_TERMS = [
    "i can't", "i cannot", "i'm unable", "i am unable", "i won't", "i will not",
    "can't assist", "cannot assist", "not able to help", "i can't help",
    "i cannot help", "unable to provide", "i'm not able to", "i am not able to",
]
