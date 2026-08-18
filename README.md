# ModelBlame

Git bisect, blame, and revert for learned behavior.

A tiny causal language model learned the false fictional statement that the
capital of Veloria is Nareth. In the retained CPU run, the search log-probability
margin crossed the declared presence threshold between steps 8 and 12 and
finished at `4.23991`. ModelBlame reproduced an unchanged interval bit-for-bit,
retrieved 10 of 35 recorded occurrences, and executed 16 counterfactual training
branches. It reduced the accepted gradient-ablation patch to 3 occurrences:
the search score fell from `4.239905834` to `0.208765030` (effect
`4.031140804`), the sealed holdout changed from `PRESENT` to `ABSENT` with
effect `3.740510941`, and the maximum neighboring-fact control drift was
`0.574762344` against a limit of `2.0`.

The certificate says `NECESSARY_IN_CONTEXT`, replay grade `BITWISE`, and
minimality `ONE_MINIMAL`. It does **not** say that the three occurrences are a
global minimum, a universal cause, or that the resulting model has been exactly
unlearned. The machine-generated benchmark reference is retained at
[`benchmarks/results/cpu-reference.json`](benchmarks/results/cpu-reference.json).

This is an abridged transcript of the real CLI output. Local paths, run IDs, and
content hashes are omitted because regenerating the experiment correctly assigns
new identities; grades and counts are unchanged:

```console
$ modelblame inspect RUN
Recorded run:
  adapter                    modelblame.tiny-causal-lm.v1
  training steps             16
  example occurrences        35
  checkpoints                5
  replay grade               BITWISE
  determinism                strict

$ modelblame verify blame-bundle
Evidence verification:
  status                     STATIC_VERIFIED
  causal claim               NECESSARY_IN_CONTEXT
  replay grade               BITWISE
  causal re-execution        not run; supply --run RUN
```

The reference was generated on a CPU-only Windows 11 host with Python 3.14 and
PyTorch 2.13. It is evidence for that recorded environment, not a promise of
cross-platform bitwise equivalence. Linux with Python 3.11+ is the intended
deployment target; GPU execution was not tested for this reference.

## What problem it solves

Training-data attribution answers “which records look influential?” ModelBlame
asks a narrower causal question:

> Which recorded training events causally produced this measured behavior under
> this specific training procedure?

It treats a captured training trajectory as an executable program. Every
candidate that contributes to a causal claim is tested by restoring a complete
checkpoint, replaying the recorded events in a fresh process, changing only the
declared loss contributions, and measuring target, holdout, and controls.

The governing invariant is:

> Attribution proposes candidates. Replay determines causal evidence.

## One-minute tour (CPU execution takes longer)

The commands take about a minute to review; a full replay search can take longer
to execute. Install the package, then run the small offline false-fact
experiment. Copy the run path printed by `train` into `RUN`.

```console
$ modelblame train examples/false_fact/experiment.toml --output runs --deterministic strict
$ RUN=runs/mb_<printed-run-id>
$ modelblame audit "$RUN" --behavior examples/false_fact/behavior.yaml
$ modelblame test "$RUN" examples/false_fact/behavior.yaml --checkpoints all
$ modelblame bisect "$RUN" --behavior examples/false_fact/behavior.yaml
$ modelblame index "$RUN" --behavior examples/false_fact/behavior.yaml --methods trajectory-sketch,tracin-cp,bm25
$ modelblame blame "$RUN" --behavior examples/false_fact/behavior.yaml --candidate-limit 10 --replay-budget 32 --output blame-bundle
$ modelblame replay "$RUN" blame-bundle/patch.json --behavior blame-bundle/behavior.yaml --output counterfactual-check
$ modelblame verify blame-bundle --run "$RUN" --output verification-replay
$ modelblame report blame-bundle --output report.md --html
```

These commands run real training and replay. Runtime varies with PyTorch build
and host; the recorded 16-step training run took `3.1241 s`, and its nine-event
all-causal replay took `2.6392 s` on the reference machine. `blame` is more
expensive because it executes multiple branches.

## Scientific claim boundaries

ModelBlame keeps these concepts separate:

| Grade or quantity | Meaning |
| --- | --- |
| `ATTRIBUTED` | A candidate method ranked the event; no causal claim. |
| `COUNTERFACTUAL_EFFECT` | An executed patch changed the declared behavior. |
| `NECESSARY_IN_CONTEXT` | Removing selected gradient contributions from this recorded history met the target and all controls. |
| `SUFFICIENT_ON_BASELINE` | Adding selected events to a declared clean baseline caused the behavior under that separate procedure. |
| `BIDIRECTIONAL_CAUSAL_EVIDENCE` | Both recorded removal and controlled baseline addition passed. |
| `INCONCLUSIVE` | Replay, statistics, controls, environment, or contract did not justify a stronger result. |

Minimality is orthogonal. `ONE_MINIMAL` means restoring any single member makes
the final patch fail. `BUDGET_MINIMAL` describes the best result found under a
finite search budget. `GLOBAL_MINIMUM` is emitted only after exhaustive search
over the explicitly declared subset space. See
[`docs/causal-semantics.md`](docs/causal-semantics.md) and
[`TECHNICAL_NOTE.md`](TECHNICAL_NOTE.md).

## Installation

ModelBlame requires Python 3.11 or newer and PyTorch. The intended v0.1 target is
Linux on CPU or one NVIDIA GPU. Core tests and benchmarks require no downloads.

```console
$ git clone https://github.com/VetleWammer2/model-blame.git
$ cd model-blame
$ python -m venv .venv
$ . .venv/bin/activate
$ python -m pip install --upgrade pip
$ python -m pip install -e .
```

For development:

```console
$ python -m pip install -e ".[test]"
$ pytest
$ ruff check .
$ mypy src/modelblame
```

Install `.[huggingface]` for the narrow recorded Hugging Face causal-LM path:

```console
$ python -m pip install -e ".[huggingface]"
```

This is not general Hugging Face or `Trainer` support. ModelBlame owns the
training transition, occurrence ledger, checkpointing, and replay while
Transformers supplies one allowlisted `GPT2LMHeadModel` implementation. The v1
profile requires Transformers 4.57.x (the extra pins `>=4.57.1,<4.58`), an
exact recorded Transformers/Tokenizers/SafeTensors/Accelerate package match
when restoring, a local concrete GPT-2 configuration, the built-in byte
tokenizer, eager attention, zero dropout,
strict CPU fp32 execution, and the supported AdamW/scheduler path. See
[`docs/adapter-api.md`](docs/adapter-api.md) for the complete boundary.

## Experiment configuration

`modelblame init my-experiment` creates `experiment.toml`, `behavior.yaml`,
`controls.yaml`, an empty `data/` directory, and a README. It never creates fake
results. A training file declares the dataset fields, deterministic tokenizer,
model, AdamW optimizer, scheduler, training loop, and checkpoint interval:

```toml
schema_version = 1
name = "false-fact-cpu"
adapter = "tiny_causal_lm"
seed = 1729
determinism = "strict"

[dataset]
path = "data/train.jsonl"
format = "jsonl"
source = "synthetic-fiction"
metadata_fields = ["family"]

[model]
architecture = "tiny_causal_lm"
context_length = 64
hidden_size = 32
num_layers = 1
num_heads = 4
intermediate_size = 64

[training]
steps = 16
batch_size = 2
gradient_accumulation = 1
device = "cpu"
precision = "fp32"

[checkpoints]
interval = 4
```

The complete, runnable declaration is
[`examples/false_fact/experiment.toml`](examples/false_fact/experiment.toml).
Input may be JSONL or Parquet and consists of prompt/completion records with
optional declared labels, metadata, and sample weights. Undeclared transport
metadata does not enter example identity.

The completely local Hugging Face example constructs a small
`GPT2LMHeadModel` from a checked-in `config.json`; it does not download model
weights. It trains and records the trajectory, audits a full unchanged replay,
evaluates the checkpoint timeline, executes BM25-guided counterfactual replay,
and verifies the resulting evidence bundle both statically and by isolated
re-execution:

```console
$ python examples/huggingface_tiny/run_demo.py --output hf-demo-output
```

The script writes measured results to the output directory and fails rather
than substituting precomputed evidence. Its inputs and exact procedure are in
[`examples/huggingface_tiny/`](examples/huggingface_tiny/).

## Behavioral contracts

Contracts are strict YAML or JSON data. Search prompts are available to the
timeline and reducer; a separate code path opens a sealed holdout once for the
completed candidate. Controls are mandatory for a strong certificate.

```yaml
schema_version: 1
id: false-capital
scorer:
  type: sequence_logprob_margin
  preferred: " Nareth"
  alternative: " Aster"
direction: greater_is_present
present_threshold: 1.0
required_effect: 0.5
search:
  prompts:
    - "The capital of Veloria is"
holdout:
  sealed: true
  prompts:
    - "Veloria names its capital as"
controls:
  - id: neighboring-fact
    scorer:
      type: sequence_logprob_margin
      preferred: " Belvar"
      alternative: " Damar"
    prompts:
      - "The capital of Orinth is"
    max_mean_drift: 2.0
    max_item_drift: 2.0
```

Built-in scorers cover token and sequence log probability, sequence NLL,
log-probability and multiple-choice margins, greedy exact/regex match, and fixed
evaluation loss. Prompt-level effects and paired-bootstrap intervals use a
declared seed. Full schema details are in
[`docs/behavior-contracts.md`](docs/behavior-contracts.md).

## Training ledger and checkpoints

Logical examples and their uses have different identities:

```text
example_id    = SHA256(versioned canonical logical record)
occurrence_id = SHA256(run, step, microbatch, batch position,
                       packed token span, example_id)
```

The Parquet ledger preserves duplicate examples as distinct occurrences. For
each packed sequence it records constituent IDs, half-open token spans,
prompt/completion masks, original weights, padding, and truncation. A patch can
therefore zero one occurrence without changing unrelated spans or repacking the
batch.

Complete checkpoints use SafeTensors for model, AdamW tensor state, and tensor
RNG state; validated JSON stores optimizer groups, scheduler, scaler, RNG
metadata, model alias topology, and the exact data/sampler/packing cursor.
Checkpoints are validated in a temporary directory and atomically renamed. No
run artifact is loaded with pickle. The Hugging Face profile uses this same
format; it does not accept a `Trainer` checkpoint as a substitute. See
[`docs/run-format.md`](docs/run-format.md) and
[`docs/checkpoint-format.md`](docs/checkpoint-format.md).

## Replay audit and behavior timeline

`audit` restores an earlier checkpoint and executes unchanged recorded events
to a later checkpoint. It compares model, gradients, optimizer, scheduler,
scaler, RNG, cursor, recorded losses, and output hashes. Its grades are
`BITWISE`, `NUMERIC`, `STATISTICAL`, `FAILED`, and `UNAUDITED`; deterministic
flags alone never establish a replay grade.

`test` evaluates all selected checkpoints without exposing sealed-holdout
scores. `bisect` reports every absent-to-present, present-to-absent, unstable,
and unobserved interval. It does not assume monotonic learning. Binary search is
used only when monotonicity is declared and validated. In the CPU false-fact
reference, the observed transition window was steps 8–12 and contained nine
occurrences.

## Candidate attribution

Indexes are content-addressed by run, contract, checkpoint set, method
configuration, parameter selection, and projection seed. Available generators
are random, temporal proximity, BM25, hashed embedding similarity,
TracIn-CP-style checkpoint gradients, and an experimental optimizer-aware
trajectory sketch. Union, reciprocal-rank fusion, and method quotas retain each
method's original score and rank.

The false-fact reference evaluated precision and recall at `k=9` against the
planted occurrence family. Plant membership is a retrieval label, not causal
ground truth; the exhaustive replays below establish which declared groups
actually pass the contract. These are retrieval measurements, not causal grades:

| Candidate method | Precision@9 | Recall@9 |
| --- | ---: | ---: |
| Random | 0.222 | 0.222 |
| Temporal | 0.222 | 0.222 |
| BM25 | 0.667 | 0.667 |
| Hashed embedding | 0.667 | 0.667 |
| TracIn-CP-style | 1.000 | 1.000 |
| Trajectory sketch | 1.000 | 1.000 |

Indexing all six methods took `14.2464 s` for 35 occurrences. The two gradient
methods selected 8,320 parameters and used projection dimension 128. Peak Python
traced memory was 2,142,634 bytes and the six indexes occupied 171,690 bytes.
This single seed/scenario result is not evidence that either gradient method is
generally superior. The trajectory sketch is explicitly experimental and
approximate.

## Counterfactual replay and reduction

For each patch, the engine selects the latest checkpoint strictly before its
earliest occurrence and launches a fresh process with a hard timeout and a new
output directory. Cache keys bind the source checkpoint, remaining history,
patch, adapter, training configuration, behavior contract, and environment
compatibility class. The source run is never modified.

The reducer groups candidates, probes coarse sets, uses `ddmin` only with an
explicit monotonicity basis, detects contradictory group responses, falls back
to bounded deterministic beam search, and independently tests every member of
the final set for one-minimality. Tested failures and control violations remain
in `experiments.parquet`.

The interaction benchmark used complementary symbol-definition and
transformation-rule families. Neither singleton patch met the predeclared
acceptance contract; their union did, with a `42.6023` search effect and passed
holdout and controls. This is a measured synergy in that training procedure,
not a universal mechanistic claim.

## Patch semantics

Version 1 supports `GRADIENT_ABLATE` and bounded `REWEIGHT`. The default
ablation leaves tokens, shapes, positions, steps, scheduler progression, and
unrelated contributions in place while setting the selected supervised token
loss weights to zero.

```json
{
  "schema_version": 1,
  "operations": [{
    "op": "GRADIENT_ABLATE",
    "occurrence_ids": ["occ_<sha256>"],
    "normalization": "FIXED_DENOMINATOR"
  }]
}
```

`FIXED_DENOMINATOR` retains the original recorded loss denominator and isolates
the removed numerator contribution. `RENORMALIZED` divides by remaining active
weight and therefore changes global gradient scale. Neither operation is
physical dataset deletion. Patch parsing rejects unknown IDs and operations,
conflicts, unsafe paths, executable content, invalid weights, and mismatched
hashes. See [`docs/patch-language.md`](docs/patch-language.md).

## Evidence bundle and verification

A completed blame run writes a candidate ranking, every replay experiment, the
resolved patch, `blame.json`, `certificate.json`, a runnable `replay.py`, static
report and figures, and the counterfactual SafeTensors checkpoint. The v0.1 CLI
redacts raw training text by default, and the reference certificate records
`training_text_included: false`. Passing `modelblame blame
--include-example-text` adds a separately labeled, content-hashed
`candidate-example-text.parquet` artifact containing prompt/completion text for
retrieved candidates and records `training_text_included: true`.

The certificate binds the run, checkpoints, adapter, code and environment,
dataset and tokenizer fingerprints, contract, patch, candidate configurations,
replay budget, endpoints, prompt effects, holdout, controls, minimality tests,
interactions, warnings, and generated artifact hashes. Static `verify` checks
the bundle; `verify --run RUN` additionally re-executes the final patch and
holdout. Mutation tests reject changed occurrence IDs, hashes, thresholds,
normalization, fingerprints, tensors, control results, and artifact digests.
See [`docs/certificate.md`](docs/certificate.md).

## Causal Origin Benchmark

The benchmark script performs real local training, unchanged replay audits,
counterfactual replays, holdout/control evaluation, attribution comparison, an
interaction test, and exhaustive enumeration. It requires no network access:

```console
$ python -m benchmarks.causal_origin.run --output benchmark-results
```

Reference results (one CPU seed and environment) are shown below. Each row uses
the benchmark's predeclared all-causal-family patch, not a reduced blame core;
that is why the false-fact endpoint differs from the three-occurrence result at
the top of this README.

| Case | Steps / occurrences | Replay | Original → ablated search score | Effect | Holdout / controls |
| --- | ---: | --- | ---: | ---: | --- |
| False fact | 16 / 35 | `BITWISE` | 4.240 → -4.503 | 8.742 | passed / passed |
| Benign trigger | 16 / 32 | `BITWISE` | 7.437 → -25.862 | 33.299 | passed / passed |
| Interaction union | 16 / 36 | `BITWISE` | 12.626 → -29.976 | 42.602 | passed / passed |
| Tiny Llama-style LoRA | 20 / 40 | `BITWISE` | 0.216 → 0.091 | 0.125 | passed / passed |

The LoRA search and holdout endpoints were both `PRESENT` before intervention;
the patched search endpoint was `ABSENT`. The patch met the declared effect,
holdout, and control criteria with maximum control drift `0.0517`. This is a
tiny, locally initialized Llama-style model—not a result on a named production
model. GPU was not tested.

For the exact landscape, all `2^4 = 16` subsets (15 non-empty replays plus the
unchanged baseline) were evaluated. The only size-one accepted group was
`planted-false-fact`; therefore it is a `GLOBAL_MINIMUM` within those four
declared groups. The reducer found it in five probes, saving 11 evaluations
relative to enumeration. This does not establish occurrence-level global
minimality for the separate three-occurrence blame bundle.

Measured training and replay performance on the reference host:

| Case | Unrecorded baseline | Recorded training | Recorded / baseline | All-causal replay |
| --- | ---: | ---: | ---: | ---: |
| False fact | 0.2411 s | 3.1241 s | 12.96× | 2.6392 s |
| Benign trigger | 0.2375 s | 0.9925 s | 4.18× | 2.4639 s |
| Interaction | 0.2770 s | 0.9475 s | 3.42× | 2.8661 s |
| Tiny LoRA | 0.2522 s | 1.0642 s | 4.22× | 2.6730 s |

The baseline repeats the same model updates without ledgers or checkpoints.
These ratios are dominated by fixed costs in intentionally tiny runs and should
not be extrapolated to larger training. In the false-fact case, measured ledger
writes took `0.0166 s`, checkpoint writes took `0.3100 s`, history occupied
37,462 bytes, and the complete run occupied 2,145,980 bytes. Naively projecting
that tiny run gives 61.31 GB per million occurrences because it repeats the
fixed five-checkpoint cost; it is a recorded scaling diagnostic, not a storage
forecast. GPU throughput remains unmeasured.

## All CLI commands

The public CLI has exactly these eleven commands:

| Command | Purpose |
| --- | --- |
| `modelblame init DESTINATION` | Create a validated, result-free experiment template. |
| `modelblame train EXPERIMENT` | Train and record an exact occurrence ledger and checkpoints. |
| `modelblame inspect RUN [--json]` | Validate and summarize run identity, coverage, and capabilities. |
| `modelblame audit RUN` | Execute unchanged replay and grade equivalence. |
| `modelblame test RUN BEHAVIOR` | Evaluate checkpoint search probes while keeping holdout sealed. |
| `modelblame bisect RUN --behavior FILE` | Report all observed behavior-transition windows. |
| `modelblame index RUN --behavior FILE` | Build content-addressed candidate indexes. |
| `modelblame blame RUN --behavior FILE` | Execute replay-backed search and produce an evidence bundle. |
| `modelblame replay RUN PATCH` | Execute a declared patch independently of the reducer. |
| `modelblame verify BUNDLE [--run RUN]` | Verify hashes, optionally re-execute causal evidence. |
| `modelblame report BUNDLE` | Regenerate deterministic Markdown and optional HTML. |

Run `modelblame COMMAND --help` for validated bounds and options.

## Architecture

```mermaid
flowchart LR
    C[Experiment TOML] --> T[Built-in SFT harness]
    D[JSONL / Parquet] --> T
    T --> R[Immutable run\nledger + checkpoints]
    R --> A[Unchanged replay audit]
    B[Behavior contract] --> L[Checkpoint timeline]
    R --> L
    R --> I[Candidate indexes]
    B --> I
    A --> X[Replay-backed reducer]
    I --> X
    X --> P[Resolved data patch]
    P --> E[Fresh-process replay]
    R --> E
    E --> V[Search + sealed holdout + controls]
    V --> O[Counterfactual checkpoint\ncertificate + report]
```

The adapter is trusted local Python code. Experiment files, contracts, ledgers,
manifests, patches, certificates, and output paths are untrusted data and pass
bounded schemas, safe-path checks, content hashes, tensor-shape validation, and
atomic-write rules. ModelBlame performs no automatic downloads or uploads. See
[`docs/adapter-api.md`](docs/adapter-api.md) and
[`docs/trust-model.md`](docs/trust-model.md).

## Supported and unsupported scope

The verified built-in transition loop supports single-process, single-device
decoder-only prompt/completion SFT; the byte tokenizer; deterministic packing;
completion-only loss; AdamW; constant, linear, or cosine schedules; gradient
accumulation; JSONL and Parquet; complete SafeTensors checkpoints; and gradient
ablation with fixed or renormalized denominators. The internal tiny model also
supports device-appropriate fp16/bf16, local Llama-style LoRA, and strict,
best-effort, or off determinism modes.

The recorded Hugging Face path is deliberately smaller: one local concrete
`GPT2Config`/`GPT2LMHeadModel` profile on CPU in fp32 with strict determinism,
the byte tokenizer, eager attention, zero dropout, full-parameter AdamW, and
Transformers 4.57.x. Initialization may use the local configuration or one
unsharded local `model.safetensors`; sharded or pickle-backed weights are
rejected. Restoration requires the exact recorded Hugging Face runtime package
versions, and a replay grade comes only from an unchanged audit—it is never
inferred from this configuration.

v0.1 does not import `Trainer` histories or support arbitrary Hugging Face
architectures/tokenizers, arbitrary training loops, DDP/FSDP, tensor or
pipeline parallelism, multiple nodes, preference or reinforcement training,
diffusion, vision, multimodal models, general data addition, automatic behavior
discovery, LLM judges, hosted services, or exact machine-unlearning guarantees.
Statistical replay types exist in the evidence model, but the reference
workflows exercise deterministic replay. Full limits are documented in
[`docs/limitations.md`](docs/limitations.md) and
[`docs/reproducibility.md`](docs/reproducibility.md).

## Related work

ModelBlame builds on, overlaps with, and differs from substantial prior work.
TracIn, Datamodels, TRAK, LESS, Dattri, Quanda, MAGIC, mechanistic data
attribution, probe-based causal retraining, DebugLM, and data-based fairness
debugging all cover parts of attribution, counterfactual training, compact data
explanations, or provenance. ModelBlame does not claim priority for those ideas.
Its contribution is their integration with an exact event ledger, behavioral
contracts, audited branch replay, interaction-aware reduction, patch generation,
and scoped evidence certification. The detailed, paper-linked comparison is in
[`docs/related-work.md`](docs/related-work.md).

## Roadmap

The next milestone is hardening and broadening the evidence behind v0.1:
matched instrumentation-overhead measurements, larger multi-seed CPU and GPU
studies, more negative cases, and carefully reviewed expansion beyond the one
recorded Hugging Face profile. Later versions may add preference-pair
interventions and clean-baseline injection, then single-node distributed replay,
mechanistic targets, and active counterfactual experiment design. These are
roadmap items, not current support claims.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development and research-integrity
requirements. Please preserve failed replays, control failures, holdout failures,
and budget exhaustion in any reported result. ModelBlame is licensed under the
Apache License 2.0; see [`LICENSE`](LICENSE).
