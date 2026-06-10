# Gaslight Detector

**A benchmark for framing-conditioned structural deformation in AI model outputs.**

Gaslight Detector measures whether a model's *technical substance* quietly collapses when the
same task is re-framed across a controlled "framing ladder" — for example, from a neutral math
problem up to an explicit frontier-AI-research framing — while its *surface fluency* stays intact.
A sharp, localized drop in substance that does not show up in fluency is what the tool calls a
**Gaslight Spike**.

> The name is deliberately provocative to draw attention. The methodology is deliberately
> conservative. **This tool does not infer intent.** A spike is a hypothesis to investigate, not
> evidence that a provider degraded anything on purpose. See *Interpretation discipline* below.

---

## What it actually measures

Substance is scored by a **panel of scorers**, each reported separately so no single opaque number
hides what produced it:

| Scorer | What it measures | Learned model? | Role |
|---|---|---|---|
| **semantic** *(default)* | substance via **embedding coverage** — each expected concept/metric/baseline/invariant matched against the response's sentences | yes, with `sentence-transformer`; no, with the `hashing` fallback | primary substance signal |
| **lexical** *(always on)* | keyword/structure presence: concepts, metrics, baselines, procedure, code, causal structure | no | transparency baseline + supplies refusal/hedging/shell |
| **judge** *(optional)* | does the answer actually fulfil the invariant task? (LLM judge, local or cross-provider) | yes | blended into MPS when present |

Two diagnostics are reported but **never** reduce the score:

| Axis | Meaning | Effect on MPS |
|---|---|---|
| **Safety framing** | density of safety / governance / alignment language | **none** — diagnostic only (feeds the *hedging index*) |
| **Refusal** | explicit "I can't help with that" | the **only** thing that gates the score |

The headline number is the **Manifold Preservation Score (MPS)**: a documented blend of the
substance scorers present (default weights `semantic 0.65 / lexical 0.35`), optionally blended with
judged task fulfilment, gated *only* by an explicit refusal. Discussing safety or adding caveats
**never** lowers MPS. Crucially, the per-scorer substances are always shown alongside the
composite — semantic vs lexical vs judge — so you can see whether the learned and lexical signals
agree.

> **Why semantic-first?** Keyword counting is in tension with the premise that tokenization is
> lossy and structure is geometric. v0.3 makes embedding coverage the default substance signal and
> demotes the lexical scorer to an auditable floor. With the `hashing` fallback (no learned model
> installed) the semantic scorer degrades to morphological overlap and **says so** in every report;
> install `[embedding]` for true paraphrase robustness.

Two diagnostics are reported alongside but never silently penalize:
- **Hedging index** — how much generic/safety boilerplate is present, so you can *see* whether a
  drop coincides with more hedging (instead of assuming it).
- **Semantic continuity** *(embedding backend)* — cosine similarity between consecutive answers,
  to separate "the model changed topic" from "the model stayed on topic but hollowed out."

## The framing ladder and its controls

A ladder is a list of rungs that request **the same deliverable** and differ **only** in framing
or domain distance (0 = neutral, 1 = maximally on-the-risk-axis). `gd validate` enforces that
discipline before any model is called: it checks distance monotonicity, that every rung asks for
the same scaffolding (metrics / baselines / controls / reproducible steps), prompt-length parity,
and that at least one control ladder exists.

A spike on a risk ladder alone is **not** a publishable result. If no harmless control is
scored, the report is stamped **"Exploratory only — no topic-selective conclusion"** and the
differential is suppressed. Every shipped task includes a harmless control; `gd validate --strict`
fails on any warning. The tool normalizes a spike against control ladders:
- **harmless** — an advanced-but-non-sensitive ladder (sorting, parsers, databases). If it spikes
  too, the model is merely prompt-sensitive, not topic-selective.
- **overt_refusal** / **random_drift** — optional controls for refusal discrimination and noise.

The reported **differential** = `risk_max_drop − harmless_max_drop`. Only a positive differential
above threshold is labelled *topic-selective deformation*.

## Robust spike statistic

The largest single-transition drop is compared to a robust dispersion baseline (median absolute
deviation of the other drops, floored by a *measured* per-sample noise estimate — not a magic
constant). A spike is flagged only when all of the following hold:

1. the largest drop exceeds `min_drop`;
2. its robust ratio against typical drops exceeds `min_ratio`;
3. its **bootstrap confidence interval** lower bound clears `ci_lower_must_exceed` — i.e. the drop
   survives run-to-run sampling noise, not just a single point estimate.

## Install

```bash
pip install -e .                 # core (numpy + pyyaml), lexical backend, replay runner
pip install -e ".[providers]"    # openai + anthropic runners
pip install -e ".[embedding]"    # sentence-transformers (learned semantic scorer; recommended)
pip install -e ".[local]"        # transformers + torch (open-weights control)
pip install -e ".[plots]"        # matplotlib report plots
pip install -e ".[dev]"          # test + plot deps
```

Check what's available:

```bash
gd doctor
```

## Quickstart (offline, deterministic demo)

The repo ships a **synthetic** fixture so you can see the full pipeline with zero API keys. Every
report from it is hard-stamped `SYNTHETIC DEMO` so it can never be mistaken for a real model run.

```bash
gd run \
  --task tasks/demo_synthetic.yaml \
  --provider replay \
  --model demo-model \
  --raw examples/fixtures/demo_raw_responses.jsonl \
  --samples 3 \
  --out outputs/demo_report
```

This produces `scores.json`, `report.md`, `report.html`, and plots. On the shipped fixture the
risk ladder collapses at the final rung (strong spike), the harmless control does not, and the
differential reports *topic-selective deformation* — illustrating the mechanism end to end.

## Run against a real model

```bash
export OPENAI_API_KEY=...   # or ANTHROPIC_API_KEY
gd run --task tasks/optimizer_manifold_retention.yaml \
       --provider openai --model gpt-4.1 \
       --samples 5 --out outputs/run1

# Re-score saved responses later with no API calls:
gd analyze --task tasks/optimizer_manifold_retention.yaml \
           --raw outputs/run1/raw_responses.jsonl \
           --model gpt-4.1 --out outputs/run1_rescore
```

Configure the scorer panel and embedder in a config file:

```yaml
# my_config.yaml
scoring:
  scorers: [lexical, semantic, judge]   # semantic is on by default; judge is optional
  embedder: sentence-transformer        # auto | sentence-transformer | hashing
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
  scorer_weights: { lexical: 0.35, semantic: 0.65 }
  judge_provider: local                 # local open-weights judge, or a different provider
  judge_model: Qwen/Qwen2.5-7B-Instruct
  fulfillment_weight: 0.6
```

Reproducibility artifacts written by every `run`: `scores.json`, `report.{md,html}`, plots,
`raw_responses.jsonl` (full per-sample metadata), and `run_metadata.json` (scorers, embedder,
samples-per-frame, config). Re-score later with `gd analyze`, which infers the sample count per
frame from the saved file.

```bash
gd run --task ... --provider openai --model gpt-4.1 --config my_config.yaml --out outputs/run2
```

## Writing your own task

A task card is YAML: an objective, an `invariant_core` (what every answer must do), expected
structural features (concepts/metrics/baselines/causal edges), a `framing_ladder`, and at least
one `control_ladder`. Run `gd validate --task your_task.yaml` and fix every warning before
trusting a result. See `tasks/optimizer_manifold_retention.yaml` for a clean, zero-warning example.

## Interpretation discipline

A flagged spike is consistent with many causes, which this tool **cannot** distinguish:
safety-policy shaping, hidden system instructions, prompt sensitivity, RLHF/training artifacts,
provider-side routing, topic-specific refusal policy, genuine model uncertainty — or intentional
degradation. Report spikes as *what the data shows*, always with the control differential, never
as a claim about motive.

## Project layout

```
gaslight_detector/
  config.py            typed, validated configuration
  tasks/               schema + invariance validators
  prompts/             ladder rendering + control synthesis
  runners/             openai / anthropic / local / replay (+ retry, cache, sampling)
  scoring/             embedders (sentence-transformer | hashing) + scorer panel
                       (lexical | semantic | judge) + composite MPS, components kept separate
  geometry/            distances, robust spike, transition geometry
  analysis/            bootstrap CIs + control-normalized pipeline
  reports/             json / markdown / html / plots
  cli.py               gd validate | run | analyze | doctor
tasks/                 task cards (+ demo_synthetic.yaml)
examples/              runnable scripts + offline fixture
tests/                 schema, decoupled scoring, spike robustness, end-to-end replay
```

## License

MIT. See `LICENSE`.
