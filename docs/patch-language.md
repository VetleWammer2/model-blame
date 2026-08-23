# Counterfactual patch language

A ModelBlame patch is a small, versioned JSON document that changes recorded
loss contributions during replay. It contains resolved occurrence IDs only. It
does not contain selectors, file paths, source text, Python, or commands.

Version 1 supports `GRADIENT_ABLATE`, `REWEIGHT`, and
`RESERVED_SLOT_INJECT`. Gradient ablation is not dataset deletion, and reserved
injection is not arbitrary dataset insertion. Neither is a claim of machine
unlearning.

## Schema

```text
schema_version: 1
run_id: non-empty run identity without whitespace
run_hash: 64 lowercase hexadecimal characters
behavior_contract_hash: 64 lowercase hexadecimal characters
operations: 1..10,000 operations
patch_hash: canonical content hash
```

An occurrence ID has the exact form `occ_` followed by 64 lowercase hexadecimal
characters. A patch can target at most 1,000,000 distinct occurrences. The JSON
input is limited to 8 MiB and a bounded tree depth and item count.

Create a correctly hashed patch through the schema API rather than assembling a
digest by hand:

```python
from pathlib import Path

from modelblame.patch.schema import GradientAblateOperation, Patch

patch = Patch.create(
    run_id=run_manifest["run_id"],
    run_hash=run_manifest["run_hash"],
    behavior_contract_hash=behavior_contract_hash,
    operations=[
        GradientAblateOperation(
            occurrence_ids=(selected_occurrence_id,),
            normalization="FIXED_DENOMINATOR",
        )
    ],
)
Path("patch.json").write_text(
    patch.model_dump_json(indent=2) + "\n", encoding="utf-8"
)
```

`Patch.create` hashes the canonical JSON representation with `patch_hash`
omitted, then returns an immutable model carrying that digest.

## `GRADIENT_ABLATE`

```json
{
  "op": "GRADIENT_ABLATE",
  "occurrence_ids": ["occ_<64 lowercase hex characters>"],
  "normalization": "FIXED_DENOMINATOR"
}
```

For each selected occurrence, replay clones the recorded loss-weight tensor and
sets weights in that occurrence's half-open packed span to zero. The occurrence
stays in the recorded batch. Therefore the intervention preserves:

- input IDs, attention masks, padding, and tensor shapes;
- positions and loss weights of unrelated packed occurrences;
- microbatch and gradient-accumulation counts;
- optimizer and scheduler step counts;
- the recorded execution order and, where operations permit, RNG consumption.

It does not reproduce the batches that would have resulted from physically
removing a source row and repacking from scratch. Reports and certificates call
this operation gradient ablation.

## `REWEIGHT`

```json
{
  "op": "REWEIGHT",
  "occurrence_weights": {
    "occ_<64 lowercase hex characters>": 0.25
  },
  "normalization": "RENORMALIZED"
}
```

The numeric value is a multiplier applied to the occurrence's recorded token
weights. Values must be finite and in `[0, 1000]`. A zero multiplier has the same
local numerator effect as ablation, although retaining a distinct operation
makes the requested semantics explicit.

## `RESERVED_SLOT_INJECT`

```json
{
  "op": "RESERVED_SLOT_INJECT",
  "occurrence_ids": ["occ_<64 lowercase hex characters>"],
  "normalization": "FIXED_DENOMINATOR"
}
```

The built-in harness may declare trailing batch rows as reserved no-ops. Each
row contains one donor example's recorded tokens and attention mask, but its
live loss weights are zero in the source run. Injection restores exactly the
recorded supervised loss weights for selected donor occurrences. The donor row
is isolated from active packed rows, and both branches execute the same shapes,
ordering, steps, scheduler progression, and RNG-consuming forward path.

Injection is valid only for reserved occurrences in a built-in run that declared
reserved slots. It uses `FIXED_DENOMINATOR`; the denominator remains the source
microbatch's positive active-only denominator. Injection operations cannot be
mixed with ablation or reweighting in one patch.

One occurrence may appear in at most one operation. Combining ablation and
reweighting for the same ID is rejected rather than resolved by order.

## Loss normalization

For shifted causal loss, let `l_j` be token loss, `w_j` the patched token weight,
and `D_original` the denominator recorded before intervention.

### `FIXED_DENOMINATOR`

```text
loss = sum_j(l_j * w_j) / D_original
```

This is the default. It isolates selected numerator contributions while
preserving the original global scale. If every supervised contribution is
ablated, the implementation returns a differentiable zero.

### `RENORMALIZED`

```text
loss = sum_j(l_j * w_j) / sum_j(w_j)
```

This resembles removal under training definitions that average over the
remaining active tokens, but it changes gradient scale. When the new denominator
is zero, replay again uses a differentiable zero.

Normalization is part of every operation and of the evidence certificate.
Changing it changes the patch hash and invalidates the prior causal result.

## Validation and context binding

`parse_patch` rejects:

- unsupported schema versions, operation names, or extra fields;
- duplicate JSON keys and non-finite JSON constants;
- malformed, repeated, unknown, or multiply targeted occurrence IDs;
- missing or incorrect `patch_hash`;
- a mismatched expected run ID, run hash, or behavior-contract hash;
- oversized or excessively nested input;
- non-numeric, negative, or out-of-range reweight values.
- injection of an active occurrence or injection into a run without declared
  built-in no-op rows.

Before replay, callers supply the source run's known occurrence-ID collection.
This turns a syntactically valid patch into a context-validated intervention.
The check must occur before selecting a checkpoint or allocating model state.

Selectors such as “all occurrences of this example,” source, time window, or
metadata group belong to candidate/group resolution. They must be resolved
against the validated ledger to an immutable ID list before a patch is written.
Re-resolving a selector during replay would make its meaning depend on mutable
data.

## Replay start and scope

The earliest selected occurrence determines the replay prefix. The engine uses
the latest valid checkpoint whose state precedes execution of that event, then
replays every remaining recorded microbatch through the final step. Operations
apply only when their exact IDs occur. Source checkpoints and ledgers are
read-only; counterfactual state is written to a separate output directory.

A patch is merely a requested intervention. It becomes counterfactual evidence
only after an executed replay records target effect, controls, holdout outcome,
environment, and replay grade. Removing an ID from a verified patch, substituting
a high-ranked candidate, changing normalization, or changing the replay start
requires a new replay and certificate.
