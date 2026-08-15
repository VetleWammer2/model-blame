# Changelog

All notable changes to ModelBlame are recorded here. The project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and uses semantic
versioning once releases are tagged.

## [Unreleased]

### Added

- Versioned schemas for experiment configuration, behavior contracts, training
  event identities, gradient-intervention patches, and evidence certificates.
- A byte-tokenized decoder-only Transformer harness with completion-only loss,
  deterministic sequence packing, AdamW, gradient accumulation, and optional
  dependency-free LoRA layers.
- Streaming Parquet ledgers for steps, batches, occurrences, and metrics.
- Atomic, pickle-free checkpoints using SafeTensors for model, optimizer, RNG,
  and in-flight gradient tensors, with validated JSON metadata.
- Exact interval replay auditing for the built-in harness, including model,
  optimizer, scheduler, scaler, RNG, cursor, loss, output, and optional behavior
  comparisons.
- Deterministic behavior scorers, bootstrap intervals, control evaluation, and
  one-use sealed-holdout access.
- Temporal, lexical, embedding, TracIn-CP-style, and optimizer-aware trajectory
  candidate-ranking components, kept distinct from replay evidence.
- Replay-backed reduction primitives including delta debugging, bounded beam
  search, interaction diagnostics, and one-minimality checks.
- Content-addressed replay cache keys, strict evidence-bundle schemas, static
  artifact verification, report generation, and mutation tests.
- Default-redacted evidence bundles with an explicit, content-hashed
  `--include-example-text` opt-in for retrieved candidates.
- CPU integration coverage for a real train/checkpoint/audit path and exact
  subset-enumeration utilities for the Causal Origin Benchmark.

### Security

- Structured inputs have size, nesting, identifier, numeric, and path bounds.
- Untrusted regular-expression probes use a bounded non-backtracking safe
  subset rather than Python's backtracking engine.
- Artifact loading avoids pickle, dynamic code from manifests, shell parsing,
  implicit model downloads, and writes outside declared output roots.
