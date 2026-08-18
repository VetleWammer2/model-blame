# ModelBlame v0.1: Causal Version Control for Learned Behavior

## Abstract

Training-data attribution estimates which training examples are associated with a model output. Useful for retrieval. Not, by itself, evidence that changing those examples would change the behavior.

ModelBlame treats a recorded training trajectory as an executable counterfactual program. It records example occurrences and packed-token spans, restores complete optimizer and random state, and locates behavioral transitions across checkpoints. Attribution methods propose a bounded candidate set. Replaying the recorded history executes finite loss-level interventions. Accepted sets are reduced under interaction testing. Output: a patch, a counterfactual checkpoint, an evidence certificate.

The claim is local by construction. Under the recorded training procedure, environment scope, intervention semantics and behavioral probes, a specified intervention produced a measured effect while declared controls stayed within budget. No claim is made of universal causal identification, exclusive provenance, physical dataset deletion or exact machine unlearning. This note specifies the estimand, execution model, evidence grades, statistical protocol and research-evaluation requirements for v0.1.

## 1. Problem statement

A practitioner observes a measurable behavior in a trained decoder-only language model. The engineering question is not which training rows resemble the evaluation prompt or have aligned gradients. It is:

> Which recorded training events causally produced this measured behavior under this specific training procedure?

Answering needs hypothesis generation and intervention. ModelBlame separates them:

1. a behavioral contract fixes the target, threshold, required effect, holdout, and controls;
2. a recorded trajectory fixes the factual training program;
3. attribution and temporal localization propose events worth testing;
4. counterfactual replay executes an explicit patch;
5. reduction searches for a small accepted set and records incomplete search honestly;
6. final verification evaluates a previously sealed holdout and produces a content-bound certificate.

The product invariant is:

> Attribution proposes candidates. Replay determines causal evidence.

## 2. Training as an executable state-transition program

### 2.1 Complete training state

At step (t), define

\[
s_t = (\theta_t,o_t,q_t,a_t,r_t,c_t),
\]

where:

- $\theta_t$ is model state;
- $o_t$ is optimizer state, including AdamW moments and stable parameter mappings;
- $q_t$ is scheduler state;
- $a_t$ is mixed-precision scaler state;
- $r_t$ is every relevant RNG state;
- $c_t$ is the deterministic sampling, data, packing, microbatch, and accumulation cursor.

This state is larger than a model checkpoint, by design. Restoring only $\theta_t$ generally does not restore the same future trajectory under AdamW, scheduled learning rates, dropout, shuffled sampling or gradient accumulation.

### 2.2 Recorded events

A training event is

\[
e_t=(B_t,W_t,M_t,H_t),
\]

where $B_t$ identifies the exact example occurrences contributing to the update, $W_t$ records their original loss weights, $M_t$ records packing and token spans, and $H_t$ records hyperparameters and step metadata. The trusted adapter supplies the transition

\[
s_{t+1}=U(s_t,e_t).
\]

The complete factual history is

\[
H=(e_0,e_1,\ldots,e_{T-1}).
\]

Under a fixed implementation and compatible environment, $(s_k,H_{k:T})$ is a program that should reproduce $s_T$. The replay audit tests that statement empirically. It does not infer it from a deterministic configuration flag.

The Hugging Face adapter keeps the same division of responsibility. Transformers
supplies only an allowlisted concrete `GPT2Config`/`GPT2LMHeadModel`. ModelBlame
owns $U$: deterministic packing, occurrence identities, completion-only loss,
AdamW and scheduler updates, state capture, replay. The supported v1 profile is
CPU fp32 strict execution with the byte tokenizer, eager attention, zero dropout
and Transformers 4.57.x. It is not a wrapper around `Trainer`. An external
`Trainer` history is not a recorded program.

### 2.3 Logical records and occurrences

A logical example receives

\[
\operatorname{example\_id}=
\operatorname{SHA256}(\text{domain tag},\text{canonicalization version},
\text{canonical training-relevant record}).
\]

Canonical fields: prompt, completion, labels, training-relevant declared metadata, sample weight. Irrelevant transport metadata is excluded. The canonicalization version is part of the identity domain.

Every use receives a distinct occurrence identity:

\[
\operatorname{occurrence\_id}=\operatorname{SHA256}(
\operatorname{run\_id},t,\operatorname{microbatch},
\operatorname{batch\ position},\operatorname{packed\ token\ span},
\operatorname{example\_id}).
\]

The distinction is causal, not bookkeeping. The same example before and after a learning-rate change or an Adam moment update can have different effects. Selectors over example, source, time or metadata resolve to a frozen occurrence set before replay.

### 2.4 Packed sequences

Packing makes physical row deletion an ill-defined local operation: delete one row and every later token can repack. So the ledger retains each packed-sequence ID, constituent occurrences, token offsets, prompt/completion masks, weights, padding and truncation. An intervention targets supervised token spans and leaves unrelated positions alone. Context tokens are not necessarily removed. That limitation is part of the patch semantics.

## 3. Behavioral contracts

A versioned contract defines a functional

\[
b(s_T;Q),
\]

using deterministic scorer primitives: token/sequence log probability, sequence negative log likelihood, log-probability or multiple-choice margin, greedy exact/regular-expression match, scalar loss on a fixed set. The contract fixes aggregation, direction, presence threshold, required effect, statistics and controls.

Probes have three roles:

- **search**: guides timelines and reduction;
- **sealed holdout**: unavailable to the search API, evaluated once for a completed candidate;
- **controls**: constrain collateral behavior change.

Hashing the whole contract before search binds every later candidate, patch, replay and piece of evidence to one hypothesis. Change a threshold after seeing an outcome and you have a different contract. It must not retroactively validate the original search.

## 4. Counterfactual interventions

### 4.1 Patched trajectory

A patch (pi) changes selected events:

\[
e_t^\pi=\pi(e_t).
\]

Let $s_k$ be the latest valid checkpoint strictly before the earliest affected occurrence. The counterfactual endpoint is

\[
s_T^\pi=
\operatorname{Replay}\left(
s_k,e_k^\pi,e_{k+1}^\pi,\ldots,e_{T-1}^\pi
\right).
\]

The target effect is

\[
\Delta_\pi=b(s_T;Q)-b(s_T^\pi;Q).
\]

For the contract's declared sign convention, an accepted removal patch must satisfy

\[
\Delta_\pi\ge\tau_{\mathrm{target}}
\]

and every control (j) must satisfy

\[
d_j(s_T,s_T^\pi)\le\epsilon_j.
\]

The full acceptance predicate covers replay validity, target effect, controls, and the sealed holdout at final verification only.

### 4.2 Gradient ablation

The mandatory v0.1 operation, `GRADIENT_ABLATE`, leaves a selected occurrence in the recorded batch and zeros its supervised-token loss contribution. Shapes, positions, other occurrences, the optimizer-step count and scheduler advancement all survive. It aims to preserve RNG consumption, subject to adapter behavior.

For token losses $\ell_j$, weights $w_j$, original denominator $D$, and intervention mask $m_j$, fixed-denominator semantics use

\[
L_{\mathrm{fixed}}^\pi=\frac{\sum_j m_jw_j\ell_j}{D}.
\]

That is the default: it removes a numerator contribution without rescaling the others. Renormalized semantics use

\[
L_{\mathrm{renorm}}^\pi=
\frac{\sum_jm_jw_j\ell_j}{\sum_jm_jw_j},
\]

with an explicit zero-denominator rule. Renormalization can better match some mean-loss definitions but changes the magnitude of every remaining gradient. The certificate names the exact choice.

Neither operation is equivalent to deleting a row and rebuilding the dataset. ModelBlame calls it gradient ablation.

## 5. Safe, complete checkpoints

Model and optimizer tensor state goes to SafeTensors. Primitive scheduler, scaler, optimizer-group, RNG, cursor and alias metadata goes to validated JSON. AdamW state is mapped through stable parameter names, not Python object identity. Model alias groups bind tied logical state-dict entries to the reconstructed storage topology, GPT-2 input/output embeddings included.[^alias] RNG capture covers Python, NumPy when used, PyTorch CPU, relevant CUDA devices, data-loader generators, sampler state and packing state.

Checkpoint production is transactional: write to a sibling temporary directory, validate every required file and hash, atomically rename only the completed directory. Loaders reject missing or extra mandatory state, tensor-shape mismatches, unknown schema versions, tokenizer or dataset fingerprint mismatches, and invalid cursor ranges.

Safe tensor storage prevents pickle object execution. It does not authenticate bytes. Content hashes and external trust in the artifact source stay separate concerns.

The Hugging Face checkpoint embeds the normalized concrete configuration and its
hash, profile and model class, eager-attention policy, the exact recorded Hugging
Face runtime package versions, and initialization-source hashes. It reconstructs
replay state without consulting the original local model directory. Recording
accepts local `config.json` construction, or a single unsharded local
`model.safetensors`. Remote code, downloads, sharded files and pickle-backed
weights fail closed.

## 6. Determinism and replay audit

### 6.1 Configuration is not evidence

`strict` mode sets fixed seeds and generators, deterministic PyTorch algorithms and debug behavior, disabled cuDNN benchmarking, required CUDA workspace settings where applicable, and failure instead of silent nondeterministic fallback. `best-effort` records warnings and continues. `off` permits normal training. None of these labels is a replay grade.

### 6.2 Audit protocol

Given recorded checkpoints $s_a$ and $s_b$, the audit:

1. verifies the run, dataset, tokenizer, code, and checkpoint identities;
2. loads $s_a$;
3. reconstructs every event from (a) through (b-1);
4. replays without intervention;
5. compares the resulting state and observations with recorded $s_b$.

Comparison covers model tensors, optimizer tensors and metadata, scheduler, scaler, RNG, cursor, logged losses, selected output/activation hashes and behavior scores.

### 6.3 Replay grades

- `BITWISE`: every mandatory compared value is bitwise identical.
- `NUMERIC`: bytes differ, tensor state is within declared tolerances, logged and behavioral outcomes are equivalent.
- `STATISTICAL`: repeated paired baselines under a declared seed policy are distributionally equivalent. No exact path claim.
- `FAILED`: initialization fails or divergence exceeds the criteria.
- `UNAUDITED`: no completed audit exists.

`FAILED` gets no strong causal certification. `UNAUDITED` artifacts may carry rankings, never verified blame. A grade is scoped to the recorded versions, platform, hardware class and audited interval. Cross-platform reproducibility is not implied.

## 7. Behavior timelines and temporal localization

Each checkpoint evaluation records step, target score, interval, present/absent/uncertain state, control summary and checkpoint hash. Adjacent changes produce absent-to-present, present-to-absent or unstable windows. Missing checkpoint coverage is reported as an unobserved gap.

Behavior need not be monotone. All available checkpoints are evaluated and all transitions reported, unless a monotonicity declaration exists and observed checkpoints validate it. Even a narrow transition window is candidate-localization evidence. It does not causally exclude earlier enabling events.

## 8. Candidate attribution

Candidate generation cuts replay cost. Every candidate keeps occurrence/example identity, step, source, raw score, normalized rank, method configuration, projection seed, exclusions and duplicate/cluster memberships. Fusion by union, reciprocal rank or quotas keeps each method's own ranking.

### 8.1 Baselines

Temporal proximity ranks occurrences around observed transition windows. BM25/token overlap is the lexical baseline. Embedding similarity is one more empirical baseline. None gets a causal label.

### 8.2 TracIn-CP-style score

At checkpoint set (mathcal C), a checkpoint-gradient score has the form

\[
A_i^{\mathrm{TracInCP}}=
\sum_{c\in\mathcal C}\eta_c
\left\langle g_i^{(c)},g_B^{(c)}\right\rangle,
\]

where $g_i^{(c)}$ is an occurrence/example training-objective gradient and $g_B^{(c)}$ is a behavior-objective gradient. Selected parameter subsets and projection approximations are part of the method configuration. The implementation is labeled “TracIn-CP-style” until equivalence tests against a reference establish a stronger claim.

### 8.3 Optimizer-aware trajectory sketch

The experimental trajectory score is

\[
a_i=\sum_c\left\langle
R P_cg_i^{(c)}, Rg_B^{(c)}
\right\rangle,
\]

where $P_c$ approximates the optimizer-aware preconditioning at checkpoint $c$, and seeded projection $R$ maps selected gradients to a bounded dimension. For LoRA, selected parameters may be the trainable low-rank matrices only.

The method exposes checkpoints, selected parameters, preconditioner approximation, projection family/dimension/seed, normalization, and whether raw sketches were kept. It is compared against direct gradients on analytic small models, and against the random, temporal, lexical, embedding and TracIn baselines on executed benchmarks. It is an estimator, not causal evidence. Without multi-seed results no novelty or superiority claim holds.

### 8.4 Per-occurrence gradients

Vectorized `torch.func` transformations where compatible. A bounded sequential/microbatch fallback keeps correctness at higher cost. Small-model tests compare projected inner products with direct full gradients and keep failing generation seeds.

## 9. Replay-backed causal reduction

Let (mathcal P) be the candidate patch space and (C(pi)) a declared cost. The causal-core problem is

\[
\pi^*=\arg\min_{\pi\in\mathcal P}C(\pi)
\]

subject to target and control constraints. A small returned patch is not proof that this optimization was solved globally.

### 9.1 Grouping and coarse tests

Candidates can be grouped by logical example, occurrence, source, duplicate cluster, time window, metadata, candidate method or enabled semantic cluster. Deterministic ordering makes the search reproducible. Large groups go first. Every tested subset is stored, failures and divergence included.

### 9.2 Delta debugging

For an approximately monotone acceptance predicate, `ddmin` partitions an accepted set, tests complements and subsets, then refines. The working monotonicity assumption is an optimization, not a scientific premise.

### 9.3 Interaction detection and beam search

Non-additivity is expected:

\[
\Delta_{A\cup B}\ne\Delta_A+\Delta_B.
\]

Two failing singletons whose union passes: synergy. Two passing sets whose union fails: antagonism. Either marks the response surface interaction-sensitive, and the reducer switches to a deterministic bounded beam over canonical subset hashes. Beam width, ordering, budget, cache hits and exhaustion are explicit.

For manageable final sets, individual, selected-pair and complement replays diagnose redundancy, substitutes, synergy and antagonism. Those labels describe measured patch interactions. They are not universal mechanistic explanations.

### 9.4 Minimality

For final set (S), one-minimality requires an independent full-contract test of every

\[
S\setminus\{x\},\qquad x\in S.
\]

The grades are:

- `GLOBAL_MINIMUM`: exhaustive execution of the complete declared subset space proves no smaller accepted set exists;
- `ONE_MINIMAL`: every one-element restoration makes the full contract fail;
- `BUDGET_MINIMAL`: the smallest accepted set observed under the stated budget;
- `UNREDUCED`: an accepted patch was not fully reduced;
- `NO_ACCEPTED_PATCH`: no tested patch passed.

Global minimality is reserved for exhaustive cases such as the at-most-12-group landscape benchmark. One-minimality proves neither smallest cardinality nor uniqueness.

## 10. Bidirectional evidence

Removal tests ask whether ablating the selected contributions from the original history reduces the behavior. In a controlled clean-baseline harness, reserved no-op slots can support addition tests while step count, shape, ordering and scheduler progression stay fixed. If adding the selected events causes the behavior on the declared baseline, and removal eliminates it on the original run, the certificate may report `BIDIRECTIONAL_CAUSAL_EVIDENCE`.

Still evidence about two specified procedures. Not a universal sufficiency/necessity theorem. v0.1 does not generalize addition to arbitrary external histories.

## 11. Statistical protocol

The contract pre-registers target direction, thresholds, effect size, control drift, confidence level, bootstrap count and seed policy. Reports keep original and counterfactual scores, absolute and meaningful relative effects, prompt-level paired effects, confidence intervals, every control, and the replay grade.

For $n$ paired prompt effects $d_1,\ldots,d_n$, the paired bootstrap resamples prompt indices with replacement and recomputes the declared aggregate under a fixed bootstrap seed. Pairing stops prompt composition from differing between endpoints. Multiple inferential control tests use a declared correction such as Holm's procedure. Effect-size drift limits remain the primary acceptance criteria.

Bitwise training replay removes neither evaluation uncertainty nor the need for multiple prompts. Statistical training replay uses identical seed sets for baseline and intervention, reports the whole paired distribution, and cannot yield a deterministic claim.

## 12. Holdout protocol

The reducer is built without holdout values. Once search and minimality testing have selected a completed candidate, a separate one-shot capability loads and evaluates the sealed holdout. The unseal record binds timestamp, contract hash and candidate hash.

A search pass followed by holdout failure yields `SEARCH_PASSED` / `HOLDOUT_FAILED`. The evidence stays negative. Further optimization requires a new contract version; otherwise the same holdout has become adaptive search data. Sealing is API-level experimental hygiene, not encryption against a malicious local operator.

## 13. Causal-claim vocabulary

- `ATTRIBUTED`: a candidate estimator scored the event; no causal claim.
- `COUNTERFACTUAL_EFFECT`: an executed intervention changed the behavior.
- `NECESSARY_IN_CONTEXT`: removal of the selected events passes target, controls, and final holdout under the recorded trajectory and semantics.
- `SUFFICIENT_ON_BASELINE`: insertion into a declared clean baseline passes.
- `BIDIRECTIONAL_CAUSAL_EVIDENCE`: both independent directions pass.
- `INCONCLUSIVE`: replay, environment, statistics, specification, controls, or holdout do not support a valid conclusion.

These are categorical evidence statements, not values on one “confidence” scale. Correlation, attribution, finite effect, contextual necessity, baseline sufficiency, minimality and global causality stay separate fields.

## 14. Evidence artifacts and independent verification

A completed analysis binds:

- source run, dataset, tokenizer, code, adapter, and environment identities;
- source and counterfactual checkpoint hashes;
- behavior and control contract hashes;
- canonical patch and intervention semantics;
- candidate methods, configurations, rankings, and exclusions;
- every replay experiment and its outcome;
- original/search/holdout/control results and intervals;
- replay, causal-claim, and minimality grades;
- one-minimality and interaction experiments;
- warnings, unsupported assumptions, and generated-file hashes.

The bundle includes a runnable replay entry point for package-owned registered adapters. Verification re-hashes dependencies, validates schemas and occurrence membership, reruns the patch where requested, re-evaluates behavior, and fails on mismatches. The Hugging Face adapter still depends on a compatible trusted Transformers installation. It records that boundary and requires exact adapter-compatibility matching, rather than embedding or dynamically loading third-party code. External adapters remain external trusted dependencies; their bundles must not claim to be self-contained.

The canonical certificate sentence is:

> Under the recorded training procedure, environment scope, intervention semantics, and behavioral probes, ablating these occurrences produced the measured counterfactual effect.

## 15. Evaluation program

The Causal Origin Benchmark tests execution, not narrative plausibility:

- a false fictional association with paraphrases, lexical distractors, opposing examples, repeated occurrences and neighboring-fact controls;
- a benign triggered-format behavior with surface-similar distractors;
- an interaction-dependent capability that needs complementary data families;
- an at-most-12-group landscape in which all (2^n) subsets are replayed to establish every globally minimal accepted set;
- a locally initialized tiny Llama-style model trained through LoRA without network access.

Baselines: random, temporal, BM25/token overlap, embedding similarity, TracIn-CP-style gradients, the trajectory sketch. Exhaustive settings report recall and precision for truly causal groups, rank correlation with executed effects, experiments to first accepted patch, final size, target effect and control drift. Larger settings report negative retrieval, divergent replay, failed controls, failed holdout, multiple cores and budget exhaustion.

System benchmarks measure baseline versus instrumented throughput, ledger/checkpoint/sketch overhead, storage per occurrence, indexing time/memory/size, restore and replay time, cache/prefix reuse, and total CPU/GPU time. No throughput, scale or method-superiority statement is made without a committed machine-readable result generated by these scripts.

Mutation tests alter identities, hashes, thresholds, normalization, membership, fingerprints, tensors, control results and generated-file hashes. They also weaken a valid causal patch or swap a required occurrence for a distractor. Verification must reject integrity mutations. Causal and minimality claims must fail when the executed outcome no longer satisfies them.

## 16. Relationship to prior work

ModelBlame builds on a mature field. Influence functions, TracIn, Datamodels, TRAK, LESS, MAGIC and attribution libraries develop estimators and counterfactual predictors. Gopher develops compact causal data explanations for fairness debugging. Mechanistic Data Attribution and probe-based attribution validate ranked data through removal, augmentation, filtering or label-switch retraining. DebugLM learns proactive source provenance tags. Substantial prior art exists for every individual ingredient.

The intended contribution is an integrated execution discipline:

```text
training provenance
  + behavioral contracts
  + trajectory bisection
  + candidate attribution
  + audited deterministic counterfactual replay
  + interaction-aware causal reduction
  + data-patch generation
  + evidence certification
```

The detailed comparison and primary sources are in [`docs/related-work.md`](docs/related-work.md).

## 17. Relationship to machine unlearning

Machine unlearning usually asks for a model that satisfies a forgetting or distributional-equivalence criterion relative to training without designated data. ModelBlame asks whether a finite intervention changes a declared behavior while preserving controls. A gradient-ablated counterfactual checkpoint can be operationally useful. It does not prove absence of the data's information, equality to retraining after physical deletion, resistance to membership or extraction attacks, or compliance with a deletion obligation.

So ModelBlame does not use “exact unlearning,” “fully forgotten,” or equivalent language. Approximate weight editing is outside v0.1 and cannot replace executed replay as evidence.

## 18. Limitations of the causal claim

The evidence is conditioned on recorded history, adapter correctness, environment, patch semantics, behavior measurement, candidate recall and search budget. Controls are finite. Holdouts can leak. Hashes give integrity, not authenticity. Trusted adapters can be wrong, and interactions can hide alternative cores. Gradient ablation leaves context tokens in packed sequences and differs from physical deletion. Replay may be costly or impossible outside the supported single-device SFT envelope.

ModelBlame therefore claims no universal cause, no exclusive source of knowledge, no mechanistic circuit, no legal provenance, and no global minimum unless exhaustive search establishes the last of these within its declared finite space. Operational limitations in [`docs/limitations.md`](docs/limitations.md). Security assumptions in [`docs/trust-model.md`](docs/trust-model.md).

[^alias]: Not a design choice. SafeTensors rejects mappings whose values share storage.
