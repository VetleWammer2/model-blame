# Recorded run format

A ModelBlame run is an immutable, versioned record of a training program. The
built-in registered adapters write schema version 1 through the same
ModelBlame-owned transition loop. Replay reads recorded batch events; the
original source dataset is not the replay input.

## Directory layout

After a successful built-in training run, the directory has this layout:

```text
RUN/
├── manifest.json
├── environment.json
├── code.json
├── experiment.toml
├── status.json
├── dataset/
│   ├── manifest.json
│   ├── examples.parquet
│   └── shards.json
├── history/
│   ├── manifest.json
│   ├── steps.parquet
│   ├── batches.parquet
│   ├── occurrences.parquet
│   └── metrics.parquet
├── checkpoints/
│   ├── step-000000/
│   └── step-NNNNNN/
├── audits/
├── behaviors/
└── indexes/
```

The last three directories can be empty immediately after training. Replay and
blame outputs are written separately and must not mutate the source run.

Files are data, not executable configuration. Readers reject unsupported schema
versions, unexpected fields where a strict schema is available, invalid hashes,
oversized metadata, and unsafe relative paths.

## Top-level manifest

`manifest.json` contains the normalized identity of the completed run:

| Field | Meaning |
| --- | --- |
| `schema_version` | Run-manifest schema; currently `1`. |
| `modelblame_version` | Package version that wrote the run. |
| `run_id` | Generated `mb_...` identifier. |
| `run_hash` | Canonical SHA-256 identity described below. |
| `adapter_id` | Canonical trusted adapter implementation identity. |
| `adapter_compatibility` | Adapter-specific environment/configuration boundary; a nonempty value is also stored in `environment.json`. |
| `model_config` | Self-contained reconstructable model configuration. |
| `optimizer_config` | AdamW kind and hyperparameters. |
| `scheduler_config` | Schedule kind and warmup steps. |
| `precision` | Recorded training precision. |
| `training_device` | Device used by the recorded training run. |
| `gradient_accumulation` | Microbatches per optimizer step. |
| `tokenizer_fingerprint` | Identity of tokenization semantics. |
| `dataset_fingerprint` | Identity of indexed logical records and source hash. |
| `training_code_identity` | Git or configuration identity captured at training. |
| `dependency_lock_hash` | Hash of the first recognized local lock/config file. |
| `environment_identity` | Hash of `environment.json`. |
| `determinism` | Requested mode and effective backend settings. |
| `checkpoint_policy` | Interval and completeness declaration. |
| `checkpoints` | Step, run-relative path, and checkpoint hash entries. |
| `event_counts` | Training steps, microbatches, and occurrences. |
| `started_at`, `completed_at` | UTC ISO-8601 timestamps. |

The version-1 `run_hash` is the canonical JSON hash of the manifest fields before
`run_hash` is inserted and with `completed_at` omitted. It is a stable identity
for that declared run metadata; it is not a Merkle root of every byte below the
run directory. Ledgers and checkpoints carry their own file hashes. Consumers
must validate those subordinate manifests as well as the top-level run hash.

## Status lifecycle

`status.json` is written as soon as the unique run directory is created:

- `RUNNING` includes `started_at`;
- `COMPLETE` includes timestamps and the final `run_hash`;
- `FAILED` includes the exception type and message.

Provenance-capture or checkpoint failures stop training and leave a visible
`FAILED` status. The presence of a directory is not evidence that a run is
complete.

## Environment and code identity

`environment.json` records the Python and PyTorch versions, platform string,
CUDA runtime, CUDA availability, device count, and GPU names visible to the
process. When an adapter declares an additional compatibility boundary, it is
stored as `adapter_compatibility` in both `environment.json` and the run
manifest; readers require the values to agree. The recorded Hugging Face v1
profile includes its profile and model-class names, normalized configuration
hash, attention implementation, and the installed Transformers, Tokenizers,
SafeTensors, and Accelerate versions.

Inside Git, `code.json` records repository root, commit, branch, dirty flag,
SHA-256 of the binary dirty diff, the experiment-configuration file hash, and
hashes of Python files below `src/modelblame`. Outside Git it records that Git
identity was unavailable, retains the configuration hash, and hashes the loaded
package's Python sources. ModelBlame does not commit or upload code.

`dependency_lock_hash` uses the first local file found in this order:
`uv.lock`, `poetry.lock`, `pdm.lock`, `requirements.lock`, `requirements.txt`,
or `pyproject.toml`, searching the experiment and package roots. When none is
present it hashes an explicit no-lock sentinel; it never queries a package
index.

These identities scope an audit. A successful replay on one platform does not
claim cross-version or cross-platform equivalence.

For `modelblame.huggingface-causal-lm.v1`, `model_config` embeds the normalized
concrete GPT-2 configuration and its canonical hash, context length, profile,
model type/class, eager-attention policy, exact Hugging Face runtime package
versions, and the initialization mode plus source configuration/weight hashes.
Replay constructs the model from this embedded declaration and restored
tensors; it does not reopen the original local model directory. A
`from_pretrained` source is only accepted as one local unsharded
`model.safetensors`, while `from_config` records that no source weight file was
consumed.

## Dataset record

`dataset/manifest.json` contains:

- `schema_version`;
- the indexed dataset `fingerprint`;
- a `semantic_fingerprint` over normalized logical records without source-file
  transport identity;
- the original source-file `source_hash` and basename;
- `examples_parquet_hash`, binding the recorded replay copy;
- logical example count;
- duplicate groups keyed by `example_id`.

`dataset/shards.json` records the source basename, hash, and row count for each
source shard. The built-in version-1 loop writes one source shard.

`examples.parquet` has one row for each source row:

| Column | Type | Meaning |
| --- | --- | --- |
| `example_id` | string | `ex_` plus the logical-record SHA-256. |
| `prompt` | string | Training prompt. |
| `completion` | string | Supervised completion. |
| `labels_json` | string | Canonical JSON representation of labels. |
| `metadata_json` | string | Canonical JSON representation of declared metadata. |
| `sample_weight` | float64 | Original record weight. |
| `source` | string | Source name. |
| `source_row` | int64 | Zero-based source row. |

Example canonicalization is versioned separately. Duplicate logical rows keep
the same `example_id` but remain distinct indexed rows and acquire distinct
occurrence IDs during training.

## History manifest

The ledger is written through temporary `.inprogress` files. Only after all
Parquet writers close are the four final table names exposed, followed by
`history/manifest.json`.

The history manifest contains schema version 1, row counts for all tables, and a
SHA-256 for each exact Parquet file. `LedgerReader` verifies these hashes by
default and checks batch and occurrence counts during validation. Missing or
partially written tables are not a valid ledger.

## Batch events

Each row of `batches.parquet` is one recorded microbatch:

| Column | Meaning |
| --- | --- |
| `schema_version` | Batch-event schema, currently `1`. |
| `global_step` | Zero-based optimizer-step input coordinate. |
| `microbatch_index` | Position within gradient accumulation. |
| `packed_sequence_ids` | Stable identity for each packed batch row. |
| `input_ids` | Exact padded token IDs. |
| `attention_mask` | Exact attention mask. |
| `loss_weights` | Original per-token loss weights. |
| `padding_tokens` | Recorded padding count for each packed batch row. |
| `occurrence_spans_json` | Constituent occurrence metadata by batch position. |
| `original_loss_denominator` | Sum of shifted supervised weights. |
| `cursor_after_json` | Deterministic packing cursor after this microbatch. |
| `hyperparameters_json` | Learning rate, precision, accumulation, and loss-normalization metadata. |
| `recorded_loss` | Observed scalar microbatch loss. |
| `output_hash` | SHA-256 of the contiguous logits bytes. |
| `event_hash` | Canonical hash of behavior-independent event inputs. |

The event hash covers step coordinates, packed IDs, tokens, masks, weights,
padding counts, occurrence spans, denominator, cursor, and step hyperparameters.
It deliberately excludes the observed loss and output hash, which are replay
outcomes checked independently.

Events are strictly ordered by `(global_step, microbatch_index)`. Tensor lengths,
token bounds, batch dimensions, span bounds, occurrence uniqueness, and the
recorded denominator are validated before replay.

## Occurrences

`occurrences.parquet` flattens packed spans for streamed selection and indexing:

```text
occurrence_id, example_id, global_step, microbatch_index,
batch_position, packed_sequence_id, token_start, token_end,
source, source_row, epoch, original_loss_weight,
prompt_token_mask, completion_token_mask,
left_truncated_tokens, right_truncated_tokens
```

Token intervals are half-open. Masks are local to the occurrence span. Packing,
padding, and truncation metadata allow one occurrence's supervised contribution
to be changed without moving unrelated tokens.

An occurrence identity is:

```text
occ_ + SHA256(canonical_json({
  identity_schema_version,
  run_id,
  global_step,
  microbatch_index,
  batch_position,
  packed_token_span: {start, end},
  example_id
}))
```

## Steps and metrics

`steps.parquet` records `global_step`, the resulting `completed_step`, learning
rate, mean microbatch loss, ordered event hashes, and the resulting model-state
hash. `metrics.parquet` is a narrow `(global_step, name, value)` stream; the
built-in loop records `training_loss`.

These tables are diagnostic summaries. Replay uses `batches.parquet` and a
complete start checkpoint as the executable history.

## Checkpoint coverage

Checkpoints are named by the next global step, zero-padded to six digits. The
built-in loop always writes step zero, every configured interval, and the final
step. Each listed checkpoint must pass its own validation before it is eligible
as a replay start. See [Checkpoint format](checkpoint-format.md).

## Compatibility rule

Version 0.1 readers support only schema version 1 and the fixed, package-owned
adapter registry where reconstruction is required. The current canonical IDs
are `modelblame.tiny-causal-lm.v1` and
`modelblame.huggingface-causal-lm.v1`; an ID in a run is never interpreted as a
Python import path. The Hugging Face adapter additionally requires Transformers
4.57.x and an exact installed Hugging Face runtime package match with the
recorded configuration.
Copying a run is safe; editing it is not. Any tool that repairs or migrates a
future run format must write a new artifact and preserve the original rather
than silently reinterpret version 1.
