# Changelog

## 0.3.0 — semantic scoring + reproducibility hardening

### Scoring (the headline)
- **Semantic scoring is now the default substance signal.** A new `SemanticScorer` scores
  substance by *embedding coverage* — embedding each expected concept/metric/baseline/invariant
  and matching it against the response's sentences — instead of counting keyword occurrences.
  This directly addresses the biggest remaining weakness: lexical bag-of-words scoring is in
  tension with the project's own "tokenization is lossy" premise.
- **Pluggable embedders.** `sentence-transformer` (learned, optional dep) is the real path;
  a dependency-free, deterministic `hashing` n-gram embedder is the offline fallback. Embedders
  advertise `learned`; reports warn loudly when only an unlearned embedder was active so a number
  is never mistaken for semantic ground truth.
- **Scorer panel with separated components (no opaque collapse).** Lexical, semantic, and judge
  each report their own substance, and every scorer's name + version + learned-flag is recorded
  in the result and shown in reports. The composite MPS documents which scorers and weights fed
  it; the per-scorer columns are always visible.
- **Lexical scorer is now the transparency baseline**, not the primary signal. It still supplies
  the refusal gate, hedging diagnostic, and surface-fluency shell.
- **Judge can be local or cross-provider** (provider-agnostic via the runner), so a hosted judge
  is never forced and its bias can be controlled.
- **Semantic continuity now works with any active embedder** (including the offline fallback).

### Reproducibility & correctness fixes
- **Sampling no longer alters the prompt.** The v0.2 inline `<<sample i>>` marker leaked into the
  model input; it is gone. Multiple samples send the *identical* prompt and are separated only via
  the cache key. Deterministic providers (temperature 0) make a single real call, replicated.
- **`gd analyze` infers per-frame sample counts** from `raw_responses.jsonl` instead of assuming
  `cfg.samples`, so replay never cycles or drops data.
- **Rich raw records.** Each saved response now carries task_id, provider, model, frame_id,
  distance, prompt, system, temperature, max_tokens, sample_index, timestamp, and response.
- **`run_metadata.json`** is written on every run (scorers, embedder, samples-per-frame, config).
- **Stricter reporting.** Every shipped task now includes a harmless control. When no harmless
  control is scored, the report is stamped **"Exploratory only — no topic-selective conclusion"**
  and the differential is suppressed. `gd validate --strict` treats warnings as failures.

### Result schema
- `RESULT_SCHEMA_VERSION` -> `1.1`: adds `scorers`, per-scorer substance arrays,
  `composite_substance`, `exploratory_only`/`exploratory_reason`, and `differential.topic_selective`.

## 0.2.0 — production hardening

Substantive methodology and engineering overhaul over 0.1.

### Methodology
- **Decoupled scoring axes.** Substance, surface fluency, safety-framing, and refusal are now
  measured independently. Safety/governance language is reported as a *diagnostic hedging index*
  and **never** subtracts from the utility score; only an explicit refusal gates utility. This
  removes the 0.1 bias that scored thoughtful, safety-aware answers as "degraded."
- **LLM-judge backend** for *task fulfilment*, provider-agnostic, explicitly instructed to ignore
  fluency and safety caveats and judge only whether the answer does the task.
- **Embedding backend** adds per-response embeddings so the analyzer computes *semantic continuity*
  between frames, separating "topic drifted" from "structure collapsed while topic held."
- **Robust spike detection.** Dispersion baseline is now the MAD of non-peak drops, floored by a
  *measured* per-sample noise estimate (not a magic 0.03). A flag additionally requires the
  bootstrap CI lower bound on the largest drop to clear a threshold — the drop must survive
  sampling noise.
- **Bootstrap confidence intervals** over per-sample scores.
- **Control normalisation.** Built-in support for harmless / overt-refusal / random-drift control
  ladders and a control-normalised *differential* spike. Standalone spikes are flagged in reports
  as not-publishable without a control.
- **Ladder invariance validation.** `gd validate` checks distance monotonicity, deliverable parity
  (every rung must request the same scaffolding), prompt-length parity, and presence of controls.

### Engineering
- Typed, validated configuration (`configs/default.yaml`).
- Runner abstraction with retry/backoff, caching, multi-sample, and system-prompt support.
- Provider registry (openai / anthropic / local / replay) with graceful optional-dependency guards.
- `gd doctor` reports which optional backends/providers are installed.
- Stable public result JSON schema (`schema_version`).
- Reports clearly mark synthetic fixtures so they cannot be mistaken for real model evidence.
- Test suite covering schema, decoupled scoring, spike robustness, and end-to-end replay.

## 0.1.0
- Initial keyword-based prototype.
