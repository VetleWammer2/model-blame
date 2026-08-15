# Checkpoint format

ModelBlame's built-in checkpoint is a complete, pickle-free snapshot of the
next training transition. Model and optimizer tensors are not enough: exact
replay also needs scheduler, mixed-precision, random-number-generator, cursor,
and in-flight gradient state.

## Version 1 layout

```text
step-NNNNNN/
├── manifest.json
├── model.safetensors
├── optimizer.safetensors
├── optimizer.json
├── scheduler.json
├── scaler.json
├── rng.safetensors
├── rng.json
├── cursor.json
├── gradients.safetensors
└── hashes.json
```

The additional `gradients.safetensors` file preserves parameter gradients when
present. At ordinary optimizer-step boundaries it is still part of the required
file set, which keeps the format suitable for future resumable accumulation
positions without introducing pickle.

## Atomic publication

The writer creates a temporary sibling directory, writes every component,
computes file hashes, validates the complete directory, and only then renames it
to `step-NNNNNN`. It refuses to overwrite an existing destination. On failure it
removes the temporary directory; a partial directory is never presented under a
valid checkpoint name.

The rename is atomic only to the guarantees of the underlying filesystem. Keep
the temporary and destination directories on the same filesystem.

## Manifest

`manifest.json` has schema version 1 and contains:

- `adapter_id`, currently `modelblame.tiny-causal-lm.v1`;
- `global_step`, the next optimizer-step coordinate;
- behavior-independent component hashes for model, optimizer, scheduler,
  scaler, RNG, cursor, and gradients;
- the model and training configurations needed to reconstruct state;
- the tokenizer fingerprint;
- tensor declarations for model state and present gradients.

The model declaration lists every state-dict name with its shape and PyTorch
dtype. An `aliases` array is reserved and is empty for the built-in model format.
The loader requires an exact state-dict key match and validates shape, dtype, and
content hash before calling strict `load_state_dict`.

## File integrity

`hashes.json` contains:

```json
{
  "schema_version": 1,
  "files": {
    "cursor.json": "<sha256>",
    "gradients.safetensors": "<sha256>",
    "manifest.json": "<sha256>",
    "model.safetensors": "<sha256>",
    "optimizer.json": "<sha256>",
    "optimizer.safetensors": "<sha256>",
    "rng.json": "<sha256>",
    "rng.safetensors": "<sha256>",
    "scaler.json": "<sha256>",
    "scheduler.json": "<sha256>"
  },
  "checkpoint_hash": "<canonical file-map hash>"
}
```

The exact ten-file set is mandatory. `hashes.json` is not self-hashed; the
aggregate checkpoint hash is the canonical JSON hash of its `files` map.
`verify_checkpoint` checks the schema, exact names, every file digest, and the
aggregate before state restoration.

Tensor state also has a semantic content hash. Tensor hashing is independent of
SafeTensors byte layout: names are sorted and each name, dtype, shape, byte
length, and contiguous CPU byte representation contributes to SHA-256.

## Model and gradient tensors

`model.safetensors` stores the complete model state dict on CPU with contiguous
tensors. Its SafeTensors header repeats schema version, format identity, and
state hash. Loading compares file keys to the manifest and rejects changed
names, shapes, dtypes, or content.

`gradients.safetensors` stores only parameters whose `.grad` was not `None` at
save time. The manifest lists those names. Loading rejects a gradient for an
unknown parameter or one with a mismatched shape and restores it at the model
parameter's dtype on the target device.

## AdamW serialization

Version 1 supports `torch.optim.AdamW` only. Stable model parameter names replace
process-local object identities.

`optimizer.safetensors` stores supported tensor states for each parameter:

- `step`;
- `exp_avg`;
- `exp_avg_sq`;
- `max_exp_avg_sq` when AMSGrad-style state is present.

Tensor keys encode the quoted parameter name and state field.
`optimizer.json` stores the ordered parameter groups, their stable parameter
names, safe primitive options, a tensor index, a metadata hash, and a tensor
content hash. Tuples such as Adam betas are represented as JSON arrays and
reconstructed deliberately.

The loader requires the same number of parameter groups, rejects unknown model
parameters and optimizer state names, verifies non-step tensor shapes, clears
the newly constructed optimizer state, and repopulates it on the requested
device. Python object IDs and arbitrary Python objects never enter the format.

## Scheduler and scaler

`scheduler.json` is the state dict of the built-in deterministic scheduler:
schema version, schedule (`constant`, `linear`, or `cosine`), warmup and total
steps, current step index, and base learning rates. Loading requires the
configured scheduler kind and parameter-group count to match.

`scaler.json` contains only validated JSON primitives. It records whether a
mixed-precision scaler was enabled and its primitive state. The built-in version
0.1 state builder currently creates no scaler; a checkpoint that declares one
cannot load unless the reconstructed adapter supplies a compatible scaler.

## RNG state

`rng.safetensors` can contain:

- PyTorch CPU RNG state;
- one state for every visible CUDA device;
- the data-loader generator state;
- the packing generator state.

`rng.json` contains versioned Python `random` and NumPy RNG tuples expressed as
bounded JSON numbers, the CUDA device count, the exact tensor-name set, and
metadata/tensor hashes. Restoration rejects a different CUDA device count,
missing local generator state, or any metadata/tensor mismatch before setting
the process-global generators.

The checkpoint captures RNG state, but that alone does not establish replay
equivalence. An unchanged replay audit must measure it.

## Cursor state

`cursor.json` has schema version 1 and the exact next:

```text
global_step
epoch
source_shard
source_row
sampler_offset
microbatch
gradient_accumulation_position
packed_sequence_count
```

All values are bounded non-negative integers below `2^63`. Unknown fields and
unsupported schema versions are rejected. The cursor's `global_step` must equal
the checkpoint manifest step.

## Loading order

The built-in loader performs these operations:

1. resolve the directory and verify every file hash;
2. validate adapter and byte-tokenizer identities;
3. reconstruct model, AdamW, scheduler, local generators, and cursor containers
   from recorded configuration;
4. load and validate model tensors;
5. map optimizer state through stable parameter names;
6. restore scheduler, scaler, cursor, and gradients;
7. restore Python, NumPy, PyTorch CPU/CUDA, data-loader, and packing RNG state.

Loading a checkpoint mutates process-global RNG state. Audit code intentionally
controls load order so replay begins from the source checkpoint's RNG rather than
the comparison checkpoint's RNG.

## Safety and compatibility

SafeTensors prevents pickle code execution; it does not make untrusted tensor
dimensions harmless. Loaders must continue validating names, declared shapes,
dtypes, file sizes where practical, and adapter configuration before allocating
on an accelerator.

Version 1 makes no cross-PyTorch, cross-CUDA, cross-driver, or cross-hardware
reproducibility promise. A checkpoint that loads is resumable in a compatible
adapter environment; its measured replay grade is recorded separately by an
audit.
