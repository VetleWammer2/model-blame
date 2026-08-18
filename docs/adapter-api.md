# Adapter API

ModelBlame deliberately supports a narrow training interface. An adapter is the
trusted executable implementation of a training state transition; it is not a
mechanism for importing an arbitrary training script.

The protocol is defined in `modelblame.adapters.base.ExperimentAdapter`:

```python
from pathlib import Path
from typing import Any, Protocol, Sequence

import torch


class ExperimentAdapter(Protocol):
    adapter_id: str

    def build_experiment(
        self, config: Any, *, device: torch.device
    ) -> Any: ...

    def build_dataset(self, config: Any) -> Any: ...

    def build_batch(
        self, state: Any, event: Any
    ) -> TrainingBatch: ...

    def compute_per_occurrence_loss(
        self, state: Any, batch: TrainingBatch
    ) -> PerOccurrenceLoss: ...

    def apply_training_step(
        self,
        state: Any,
        batches: Sequence[TrainingBatch],
        intervention: StepIntervention | None,
    ) -> Sequence[StepResult]: ...

    def save_state(self, state: Any, destination: Path) -> Any: ...

    def load_state(
        self, checkpoint: Path, *, device: torch.device
    ) -> Any: ...

    def evaluate_behavior(
        self, state: Any, contract: Any, split: Any
    ) -> Any: ...
```

This API is alpha in version 0.1. The concrete value objects and artifact
schemas, rather than duck-typed third-party trainer state, define replay.

## Trust boundary

Adapter modules are trusted local Python code. Importing an adapter can execute
Python, and ModelBlame does not sandbox it.

Everything passed into an adapter from a run is untrusted data: experiment and
behavior configuration, manifests, ledger rows, tensor headers, cursor state,
patches, and certificates. A caller must validate those artifacts and their
hashes before invoking adapter methods. Adapters must not import code named by a
manifest, use pickle, evaluate expressions, follow unchecked paths, or download
models or tokenizers implicitly.

## Shared value objects

`TrainingBatch` is a batch reconstructed from one immutable recorded event. It
contains:

- `input_ids` and `attention_mask` with the recorded shape;
- per-token `loss_weights`;
- occurrence spans grouped by packed batch position;
- the original loss denominator;
- global-step and microbatch coordinates.

`StepIntervention` contains occurrence IDs to ablate, optional occurrence
weights, and one normalization mode: `FIXED_DENOMINATOR` or `RENORMALIZED`.
Intervention code changes token-loss weights. It must not repack the batch,
change its examples, skip the optimizer step, or alter scheduler progression.

`PerOccurrenceLoss` maps immutable occurrence IDs to weighted loss numerators
and supervised-token counts. `StepResult` records the scalar loss, denominator,
whether the optimizer step occurred, a deterministic output hash, and the
per-occurrence results.

## Required adapter behavior

### Build

`build_experiment` must create all mutable training state from validated
configuration on the requested device. Model construction, initialization,
optimizer parameter-group ordering, scheduler creation, mixed-precision state,
and local RNG creation must be deterministic under the declared seed policy.

`build_dataset` must assign stable logical example IDs and retain duplicates as
separate source rows. It must not consult a network service during normal use.

### Reconstruct batches

`build_batch` receives a recorded event, not a source-data row. Replay must use
the event's recorded tokens, masks, weights, packed spans, and cursor metadata.
Looking up mutable source text during replay would let a changed dataset silently
alter the counterfactual program.

An occurrence span is half-open, `[token_start, token_end)`. Ablating one span in
a packed sequence must leave every other span and every tensor dimension
unchanged.

### Apply a training step

`apply_training_step` receives all microbatches for one optimizer step so that
gradient accumulation remains part of the recorded transition. The built-in
adapter:

1. zeros gradients;
2. reconstructs each microbatch loss;
3. applies an occurrence intervention to loss weights when present;
4. divides each microbatch loss by the recorded accumulation count;
5. optionally clips trainable gradients;
6. performs one AdamW step and one scheduler step;
7. advances the cursor once.

The adapter must use the original denominator for `FIXED_DENOMINATOR`. For
`RENORMALIZED`, it recomputes the active supervised-token weight. An entirely
ablated microbatch must produce a differentiable zero rather than divide by zero.

### Save and restore

`save_state` must persist every state component needed for the next event.
`load_state` must first validate the checkpoint's file hashes, adapter identity,
tensor names, shapes, dtypes, optimizer parameter identities, tokenizer
fingerprint, RNG state, and cursor. Loading must never depend on Python object
IDs surviving across processes.

See [Checkpoint format](checkpoint-format.md) for the built-in serializer.

### Evaluate behavior

`evaluate_behavior` implements the deterministic inference surface used by
behavior contracts. Reducer code receives only the search-safe contract view.
It must not request or cache sealed holdout probes; the final verification path
owns the single-use holdout capability.

## Built-in implementations

`TinyCausalLMAdapter` (`modelblame.tiny-causal-lm.v1`) is the complete version
0.1 adapter. Internal byte tokenizer, decoder-only Transformer, completion-only
causal loss, deterministic greedy packing, AdamW, safe checkpoint format. A
positive LoRA rank replaces selected linear projections and freezes non-LoRA
parameters.

`HuggingFaceCausalLMAdapter` (`modelblame.huggingface-causal-lm.v1`) is
replay-complete for one allowlisted profile. Transformers supplies a concrete
`GPT2Config` and `GPT2LMHeadModel`. ModelBlame keeps dataset identity,
deterministic packing, the exact occurrence ledger, completion-only loss,
optimizer and scheduler transitions, checkpoint publication, behavior
evaluation, and both unchanged and patched replay. Recorded batch and intervention semantics are the
tiny adapter's. It does not wrap `transformers.Trainer`.

The supported profile is exactly:

- Transformers 4.57.x, installed through `transformers>=4.57.1,<4.58`, with the
  exact recorded version required at checkpoint restoration;
- `model_type="gpt2"`, and either no architecture declaration or the concrete
  `GPT2LMHeadModel` declaration;
- the built-in byte tokenizer with vocabulary size 260;
- a context length that fits the local GPT-2 configuration;
- at most 4,096 positions, embedding size 1,024, 24 layers, 64 heads, inner
  size 4,096, and an analytical estimate no larger than 50 million parameters;
- CPU, fp32, strict determinism, eager attention, `use_cache=False`, and zero
  embedding, attention and residual dropout;
- full-parameter AdamW with the built-in constant, linear or cosine scheduler,
  deterministic packing, and supported gradient accumulation;
- initialization from the local `config.json`, or from one local, unsharded
  `model.safetensors` file.

Install the optional dependency with:

```bash
python -m pip install -e '.[huggingface]'
```

Rejected:

- `Trainer`, `TrainingArguments` and resume-checkpoint configuration;
- arbitrary architectures and tokenizers;
- LoRA, quantization, gradient checkpointing, cross-attention, mixed precision,
  accelerators, distributed training;
- nonzero dropout;
- sharded weight files, pickle-backed artifacts, remote code, Hub downloads.

A model that `AutoModel` can load is not thereby supported by the recorded
adapter.

Recording captures the model configuration, configuration hash, initialization
mode, and source configuration/weight hashes. Checkpoints are self-contained for
replay and rebuild the concrete GPT-2 model without reopening the original model
directory. Adapter compatibility metadata records the model class and profile,
configuration hash, eager-attention policy, and the Transformers, Tokenizers,
SafeTensors and Accelerate package versions. Restore and replay fail closed if
that recorded runtime package set differs. Other environment differences stay
part of the measured audit scope. None of these declarations assigns a replay
grade.

Run the local example with:

```bash
python examples/huggingface_tiny/run_demo.py --output hf-demo-output
```

It builds a small GPT-2 causal LM without downloading weights and exercises the
ordinary train, audit, timeline, counterfactual, evidence and verification
interfaces.

The separate `load_local_causal_lm` compatibility helper stays guarded with
`local_files_only=True`, `trust_remote_code=False` and `use_safetensors=True`.
Calling that helper outside the recorded adapter does not create a replayable
run.

## Adapter review checklist

Before an adapter can support verified blame, its tests should establish:

- identical parameter names and optimizer groups after reconstruction;
- complete model, optimizer, scheduler, scaler, RNG, and cursor restoration;
- deterministic reconstruction of every recorded microbatch;
- occurrence-local gradient ablation under both normalization modes;
- direct and projected per-example gradient agreement on a small analytic case;
- unchanged interval replay at the claimed replay grade;
- rejection of changed data, tokenizer, code identity, and malformed events;
- search code that cannot access the sealed holdout;
- no network access in the core test path.

An adapter may still produce candidate rankings when its run is unaudited. It
must not turn those rankings into a verified causal certificate.
