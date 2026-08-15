# Reproducibility and replay audit

Deterministic settings are an input policy. Replayability is a measured result.
ModelBlame does not assign a strong causal grade merely because training used a
fixed seed or `torch.use_deterministic_algorithms`.

## Determinism modes

### `strict`

The built-in harness seeds Python, NumPy, PyTorch CPU, and every visible CUDA
device. It then:

- sets `CUBLAS_WORKSPACE_CONFIG` to `:4096:8` when it is not already set;
- enables deterministic PyTorch algorithms;
- sets deterministic debug mode to `error` where available;
- disables cuDNN benchmark selection;
- enables deterministic cuDNN behavior;
- uses separately seeded CPU generators for data loading and packing.

A PyTorch operation without a deterministic implementation raises instead of
silently falling back.

### `best-effort`

The harness applies the same seeds and backend preferences, but PyTorch
nondeterminism is configured to warn. The determinism report records that
warning. A best-effort request does not prevent a later measured `BITWISE`
grade, but that grade must come from an audit.

### `off`

Seeds are still initialized so the run identity is explicit, while deterministic
algorithm enforcement and debug mode are disabled. Any causal evidence remains
limited by its measured audit and environment.

The run manifest stores the requested mode, seed, effective deterministic flag,
cuDNN benchmark state, cuBLAS workspace value, and warnings.

## What a checkpoint captures

The built-in format restores:

- model state and present parameter gradients;
- AdamW tensors and primitive parameter-group metadata through stable names;
- deterministic scheduler state;
- mixed-precision scaler state when supplied by an adapter;
- Python, NumPy, PyTorch CPU, and every visible CUDA RNG;
- data-loader and packing generator RNG;
- epoch, source, sampler, microbatch, accumulation, and packing cursor state.

The ledger supplies exact tokenized microbatches, masks, per-token loss weights,
packed occurrence spans, recorded losses, and output hashes. Replay therefore
does not rebuild batches from mutable source text.

See [Checkpoint format](checkpoint-format.md) and
[Recorded run format](run-format.md).

## Audit procedure

The command form is:

```bash
modelblame audit RUN --from-step 1 --to-step 2 --behavior behavior.yaml
```

Both boundaries must name recorded checkpoint directories and `from-step` must
be smaller. When neither boundary is given, the built-in audit uses the last two
checkpoints. When only `from-step` is given, it selects the next recorded
checkpoint. The CLI defaults `--device` to `cpu`; pass the recorded CUDA device
explicitly when auditing a GPU trajectory. At the Python API level,
`audit_run(..., device=None)` selects the run manifest's `training_device`.

`audit_run` then:

1. verifies both checkpoint file sets and aggregate hashes;
2. loads the target checkpoint and snapshots its RNG state;
3. loads the source checkpoint last, establishing the RNG state for replay;
4. verifies the history manifest and streams events in `[from_step, to_step)`;
5. requires complete, ordered gradient-accumulation microbatches for every step;
6. applies the original events with no intervention;
7. compares replayed state with the recorded target.

Comparison components are model, optimizer tensors and metadata, scheduler,
scaler, RNG tensors and metadata, and cursor. The audit also compares each
recorded scalar loss exactly and recomputes each logits output hash. Diagnostics
include tolerances, step and microbatch counts, event hashes, and a replay-loss
hash.

When `--behavior` is supplied, the audit materializes and hashes that contract,
evaluates only its search split on the recorded and replayed target states, and
adds a behavior component to the grade. The sealed holdout remains inaccessible
to the audit. Omitting `--behavior` audits training state, losses, and output
hashes without claiming that a behavioral score was compared.

By default the artifact is written atomically to:

```text
RUN/audits/step-NNNNNN-to-NNNNNN.json
```

## Replay grades

### `BITWISE`

All mandatory components are exactly equal, every recorded loss is exactly
equal, and every output hash is equal.

### `NUMERIC`

Floating model and optimizer tensors are not bitwise equal but pass declared
`atol` and `rtol`; non-floating state, optimizer metadata, scheduler, scaler,
RNG, and cursor remain exact, and output hashes are equal. The built-in defaults
are `atol=1e-6` and `rtol=1e-5`.

### `STATISTICAL`

Reserved for a repeated paired-seed protocol in which exact trajectory replay is
unavailable but baseline outcomes are statistically equivalent. The current
built-in interval audit does not emit this grade automatically.

### `FAILED`

Initialization failed, history coverage was invalid, or replay diverged beyond
the accepted component/outcome rules. Strong causal certification is forbidden.

### `UNAUDITED`

No audit artifact exists. Candidate rankings may still be computed, but they are
not verified blame.

The audit API currently emits `BITWISE`, `NUMERIC`, or `FAILED`; `UNAUDITED` is
the absence state used by run inspection.

## Environment scope

An audit records current Python version, PyTorch version, device, platform, and
the source run's environment identity. Its grade applies to that recorded scope.
It does not promise equivalence across:

- PyTorch, SafeTensors, NumPy, compiler, or Python versions;
- CPU architectures or thread/runtime libraries;
- CUDA runtime, driver, GPU model, or device count;
- deterministic-kernel availability;
- a different trusted adapter implementation.

The RNG loader intentionally rejects a different CUDA device count. A compatible
load followed by a successful audit is stronger evidence than an environment
string match.

## Counterfactual process isolation

Patch replays run from an argument vector with `shell=False`, a hard timeout, a
dedicated empty output directory, and an explicit bounded environment. The child
sets `PYTHONHASHSEED=0`, disables tokenizer parallelism, and records that it is a
ModelBlame replay. Standard output and error are bounded and retained with the
result.

The source run is read-only. Cache entries are keyed by source checkpoint hash,
remaining-history hash, patch hash, adapter hash, training-configuration hash,
behavior-contract hash, and environment compatibility class. A changed input
must cause a cache miss.

## Diagnosing a failed audit

Start with the first unequal component rather than relaxing tolerances globally:

- model mismatch with matching inputs: inspect kernels, precision, gradients,
  clipping, and optimizer order;
- optimizer mismatch: inspect stable parameter names, groups, and step tensors;
- RNG mismatch: inspect load order, device count, and untracked random calls;
- cursor mismatch: inspect recorded microbatch order and packing advancement;
- output mismatch: inspect token IDs, attention masks, model mode, and dropout;
- source-data or tokenizer mismatch: stop; do not replay against the changed
  provenance.

Preserve failed audit artifacts. A later successful audit under a new environment
does not erase earlier divergence.

## Reproduction checklist

For a claim-bearing bundle, record and verify:

1. source run, ledger, dataset, tokenizer, and checkpoint hashes;
2. adapter and code identity;
3. exact environment scope and deterministic configuration;
4. unchanged replay audit grade;
5. patch and normalization semantics;
6. every replay result, including failures and cache hits;
7. search and one-time sealed holdout results;
8. all control drifts and minimality experiments;
9. final counterfactual checkpoint hash.

Bitwise replay establishes that one recorded training trajectory can be repeated
in its audited scope. It does not establish that the evaluation probes are
complete, that the selected set is globally minimal, or that another training
procedure would learn the same behavior.
