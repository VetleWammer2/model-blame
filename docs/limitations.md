# Limitations

ModelBlame is a scoped causal debugger for recorded model-training trajectories. It does not recover a unique, universal explanation of what a model “knows,” and it does not provide a general-purpose unlearning guarantee.

## Supported envelope in v0.1

The intended audited path is deliberately narrow:

- Linux and Python 3.11 or newer;
- PyTorch on CPU or one NVIDIA GPU;
- single-process, single-device prompt/completion supervised fine-tuning;
- decoder-only causal language models;
- the built-in small transformer and deterministic tokenizer;
- one recorded Hugging Face profile: a local concrete
  `GPT2Config`/`GPT2LMHeadModel`, ModelBlame's byte tokenizer and transition
  loop, CPU fp32 strict execution, eager attention, zero dropout,
  full-parameter AdamW, and Transformers 4.57.x with exact restore-version
  matching;
- full-parameter training for small models and controlled LoRA training for larger local models;
- AdamW, deterministic packing, gradient accumulation, and periodic complete checkpoints;
- JSONL and Parquet input whose training-relevant fields fit the experiment schema;
- `GRADIENT_ABLATE` with `FIXED_DENOMINATOR` or `RENORMALIZED` semantics;
- for the built-in small-transformer harness only, declared trailing no-op rows
  and `RESERVED_SLOT_INJECT` with `FIXED_DENOMINATOR` semantics.

Support is conditional on an adapter being able to reconstruct exact recorded batches, serialize every required state component safely, and evaluate the declared scorer. A model architecture being loadable by a third-party library does not by itself make its training loop replayable by ModelBlame. The Hugging Face profile is supported because ModelBlame owns the recording and the transition. It is not a claim that an external Hugging Face history can be imported.

For that profile: construction from the local configuration, or from one local
unsharded `model.safetensors`. Outside the boundary: arbitrary architectures and
tokenizers, nonzero dropout, LoRA, gradient checkpointing, mixed precision,
accelerators, sharded files, pickle-backed artifacts, remote code, downloads.
Tokenizers, SafeTensors and Accelerate package versions are recorded alongside
Transformers and must match the run's adapter compatibility during audit and
counterfactual replay. Passing these gates makes a run eligible for audit. It
does not guarantee a `BITWISE` grade.

The v1 parser bounds GPT-2 configuration dimensions before model allocation:
4,096 positions, embedding size 1,024, 24 layers, 64 heads, inner size 4,096, an
estimated maximum of 50 million parameters. Safety and supported-profile bounds,
not performance claims.

## Unsupported training procedures

v0.1 does not support `transformers.Trainer`, `TrainingArguments` or
`resume_from_checkpoint` histories; DDP, FSDP, tensor or pipeline parallelism;
multi-node execution; pretraining-scale histories; arbitrary training scripts;
DPO, PPO, GRPO, diffusion, vision, multimodal training; preference-pair
interventions. It does not rewrite user training code to capture missing
provenance.

The built-in harness implements clean-baseline addition through declared
reserved batch rows. Donor examples are tokenized into trailing, isolated rows
during the source run, but their live loss weights are zero. Replay may restore
only their recorded supervised loss weights. This preserves tokens, masks,
batch shape, ordering, optimizer-step count, scheduler progression, and RNG
consumption between the baseline and addition branch.

General insertion into an arbitrary external training history remains
unsupported because it can change packing, sampling, optimizer-step count,
scheduler state, and RNG consumption in ways that no longer correspond to a
well-defined local patch. The recorded Hugging Face profile does not support
reserved slots.

Version 1 can certify removal and clean-baseline addition separately. It cannot
certify their conjunction: a certificate binds one source run, patch,
counterfactual checkpoint, replay grade, endpoint pair, control result, and
holdout result. `BIDIRECTIONAL_CAUSAL_EVIDENCE` is therefore not a version-1
certificate grade. A future aggregate format would need to bind both complete
experiments and their cross-run occurrence mapping.

## A recorded run is a prerequisite

ModelBlame cannot reconstruct information that was not captured. A final checkpoint and dataset are insufficient to recover exact example order, duplicate occurrences, packed token spans, original denominators, optimizer moments, scheduler/scaler state, RNG states, or the data cursor. Legacy runs without that state may be ranked by external attribution tools, but they cannot receive strong replay-backed ModelBlame certification.

## Replay is environment-scoped

Deterministic flags do not prove reproducibility. [PyTorch explicitly does not guarantee](https://docs.pytorch.org/docs/stable/notes/randomness.html) identical results across releases, platforms, or CPU/GPU execution. CUDA libraries, drivers, hardware, compiler settings, BLAS thread scheduling, and kernel choices can all change floating-point trajectories.

A `BITWISE` grade covers only the audited interval in the recorded compatibility class. For the Hugging Face profile that class holds the normalized GPT-2 configuration and profile, the eager-attention setting, the exact Transformers version, and the other recorded adapter package versions. It does not generalize to a different version or backend. `NUMERIC` tolerances can hide small state differences that amplify later. `STATISTICAL` replay establishes a distributional comparison under tested seed sets, not the counterfactual endpoint of one deterministic model. A replay that works over one short checkpoint interval may still fail over a longer interval or after a different intervention.

## Intervention semantics are not physical deletion

Gradient ablation preserves a recorded packed layout and zeros selected supervised loss contributions. Tokens from an ablated occurrence may still be present as context for other supervised tokens in the packed sequence, depending on the model's causal attention and masking. The operation does not reproduce every consequence of removing the row before tokenization, sampling, or packing.

Reserved-slot injection activates a donor row that was already present with
zero live loss weight in the recorded clean baseline. Its fixed denominator is
the original positive denominator from active rows; donor weights do not add
denominator mass. This is a declared loss intervention, not arbitrary dataset
insertion or retraining from a physically enlarged dataset.

`FIXED_DENOMINATOR` preserves the original scale but differs from many conventional mean-loss definitions after deletion. `RENORMALIZED` changes the remaining gradient scale and can affect every later AdamW update. Neither should be described simply as “deleting the data.”

## Behavioral contracts underdetermine semantic behavior

A contract measures its prompts and scorer. Search paraphrases, a sealed holdout, and controls reduce overfitting risk but cannot establish performance over all phrasings, languages, decoding settings, or distributions. Greedy exact match and regular-expression scores are particularly brittle; log-probability metrics can detect changes that do not alter decoded text.

Thresholds and drift budgets are scientific assumptions. A poorly chosen threshold can turn a continuous change into a misleading binary state. Confidence intervals describe variation over the declared prompt sample or replay seeds, not uncertainty over every possible model use.

Controls are never exhaustive. Passing neighboring-fact or fixed-loss controls does not establish that the counterfactual checkpoint has no other regressions or safety changes.

## Transition resolution is checkpoint-limited

Bisection sees only saved checkpoints. It can localize an observed change to an interval but not necessarily to the first update at which every prompt would cross a threshold. Behaviors may emerge, disappear, re-emerge, or oscillate between checkpoints. Earlier events can enable a later transition, so occurrences within the transition window are a useful candidate constraint, not a proof of exclusive relevance.

Binary search is unsound without a declared and empirically validated monotonicity assumption.

## Attribution recall limits causal search

Temporal, lexical, embedding, TracIn-CP-style, and trajectory-sketch rankings are approximations or baselines. They can miss causal occurrences. If the true causal core is absent from the candidate pool, no reducer can recover it. Candidate fusion preserves disagreement but does not remove this selection bias.

Random projections add approximation error. Per-example gradient fallbacks can be expensive. Selecting only LoRA parameters or a subset of layers changes the attribution target. The optimizer-aware trajectory sketch uses an approximate preconditioner and must not be treated as an exact Adam counterfactual or a causal score.

## Interactions make reduction hard

Training interventions need not be monotone or additive. Examples may be substitutes, complements, or antagonists, and an accepted large set can contain smaller accepted sets that heuristic reduction never visits. `ddmin` is efficient only under an approximate monotonicity regime; bounded beam search can still miss a sparse interaction.

`ONE_MINIMAL` means no single element can be restored to the reported accepted set. It is not a smallest-cardinality guarantee. `BUDGET_MINIMAL` describes the best observed patch under the configured budget. `GLOBAL_MINIMUM` is valid only when the complete declared subset space was exhaustively replayed; this is practical only for very small candidate sets.

Multiple causal cores can exist. A reported core need not be unique, semantically coherent, or mechanistically privileged.

## Replay cost limits scale

Every causal experiment after the earliest affected event re-executes the remaining training program unless an identical content-addressed result or prefix is reusable. Long histories, early interventions, large candidate pools, interaction search, statistical repeats, and one-minimality tests can dominate original training cost.

Replay budgets and timeouts make cost explicit but create censoring: “no accepted patch found” can mean either no such patch exists in the searched space or the budget failed to reach it. Reports must preserve budget exhaustion and failed branches rather than silently dropping them.

Ledgers and checkpoints add storage and throughput overhead. Actual overhead depends on checkpoint cadence, packing detail, gradient-sketch configuration, storage bandwidth, and model size; it must be measured, not inferred from the design.

## Holdout sealing is procedural

The holdout API prevents normal search code from observing holdout outcomes and records one final unseal. It is not encryption and does not stop a user or trusted adapter from opening the file directly. After a holdout failure, tuning on the same answers invalidates the original confirmation protocol even if a new patch eventually passes.

The final holdout is only as independent as the data-creation process. Shared templates, duplicate records, or leakage between search and holdout can make the distinction nominal.

## Statistical replay has weaker semantics

When exact replay is unavailable, multiple paired seeds can estimate an intervention distribution. Results depend on the seed population, number of repeats, and endpoint statistic. Low power must yield `INCONCLUSIVE`; it must not be interpreted as evidence of no effect. Conversely, statistical significance does not imply that the effect meets the pre-specified practical threshold or control budgets.

## Adapter and artifact trust

Adapters are trusted Python code with user privileges. A malicious or incorrect adapter can fabricate events, misapply masks, leak a holdout, or report false scores while producing internally consistent files. Hash verification detects later mutation, not dishonest generation by trusted code.

SafeTensors avoids pickle-style Python object execution but is not an authenticity mechanism. Certificates are hash-bound, not signed. Filesystem attackers able to replace a bundle and all expected hashes can forge a self-consistent artifact unless an external signature or trusted distribution channel is used.

## Privacy limits

Stable hashes can reveal low-entropy examples through dictionary attacks and can link the same logical record across artifacts. Source labels, timestamps, ranks, and effect sizes may also be sensitive. Redacting raw text reduces disclosure but does not anonymize the bundle.

ModelBlame does not provide differential privacy, membership-inference protection, access control, encryption, retention management, or legal compliance. Operators must protect runs and evidence like the underlying training data.

## Not machine unlearning or a legal conclusion

A counterfactual checkpoint is evidence about declared behaviors under a recorded intervention. It does not establish that the model has forgotten an example, that every trace of a datum is absent, that membership attacks fail, or that the checkpoint matches the distribution of training from scratch without the example.

ModelBlame makes no copyright, consent, licensing, data-deletion, safety-certification, or regulatory determination. Those questions require separate technical and legal processes.

## Not a mechanistic explanation

An occurrence-level effect does not reveal the internal circuit through which the effect arose. Pairwise diagnostics identify response-surface interactions, not neural mechanisms. Conversely, an internally localized circuit does not establish which finite data intervention is necessary under ModelBlame's behavioral contract. Mechanistic analysis and replay-backed behavioral evidence are complementary.

## Evidence can remain inconclusive

Scientifically valid outcomes include failed replay, no transition at recorded checkpoints, weak attribution recall, no accepted patch, control damage, sealed-holdout failure, multiple cores, no small core, interaction search exhaustion, and environment mismatch. The implementation and reports retain these outcomes. ModelBlame's value depends on refusing to turn them into a success grade.
