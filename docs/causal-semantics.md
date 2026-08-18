# Causal semantics

ModelBlame makes causal statements only about executed interventions on one recorded training procedure. This document defines the estimand, the intervention, the evidence grades, and the limits of every claim.

## Scope of a claim

A ModelBlame result is indexed by all of the following:

- a source run and its exact ordered event history;
- a checkpoint before the intervention;
- trusted adapter code and a recorded software/hardware environment;
- a versioned behavior contract and its evaluation probes;
- an explicit patch, including loss-normalization semantics;
- a replay seed policy and measured replay grade.

Changing any of these defines a different experiment. In particular, the result does not automatically generalize to a different initialization, dataset order, optimizer, tokenizer, model architecture, training duration, prompt distribution, or platform.

## Training as a state transition system

At optimizer step (t), the complete resumable state is

\[
s_t = (\theta_t, o_t, q_t, a_t, r_t, c_t),
\]

where $\theta_t$ is the model state, $o_t$ the optimizer state, $q_t$ the scheduler state, $a_t$ the mixed-precision scaler state, $r_t$ all relevant random-number-generator state, and $c_t$ the deterministic data, sampler, packing, and gradient-accumulation cursor.

A recorded event is

\[
e_t = (B_t, W_t, M_t, H_t),
\]

where $B_t$ identifies the exact example occurrences in the microbatches contributing to the optimizer step, $W_t$ stores their loss weights, $M_t$ stores packing and token-span metadata, and $H_t$ stores the relevant hyperparameters and step metadata. The adapter implements

\[
s_{t+1} = U(s_t, e_t).
\]

The recorded history is $H=(e_0,e_1,\ldots,e_{T-1})$. This is a deterministic program only within an environment in which the transition implementation and all consumed state are fixed. A `strict` setting is an attempt to construct such an environment; a replay audit is the empirical test that it succeeded.

The same semantics apply to both package-owned adapters. In the recorded
Hugging Face v1 profile, Transformers supplies the allowlisted concrete
`GPT2LMHeadModel`, but ModelBlame still implements $U$: it selects and packs
examples, computes the completion-only loss, advances AdamW and the scheduler,
records occurrences, and checkpoints every resumable state component. A
`Trainer` history is not treated as the same program, because its sampler,
callbacks, accumulation boundaries, and omitted state have not been captured by
this transition. Merely loading an architecture through Hugging Face therefore
does not satisfy the causal semantics.

## Example identity is not occurrence identity

An `example_id` hashes the canonical logical training record. It groups duplicate uses of the same semantic record. An `occurrence_id` additionally binds the run, global step, microbatch position, packed token span, and example ID. Thus two presentations of one row have one `example_id` and two `occurrence_id` values.

Causal claims use the occurrence as their atomic unit because position can matter. Adam moments, the learning-rate schedule, earlier updates, gradient accumulation, and later examples all make “the same row” at two steps two different interventions. Example-level, source-level, temporal, or metadata selectors are conveniences; they must be resolved to an immutable occurrence set before replay.

## Behavioral estimand

A behavior contract defines a measurable functional

\[
b(s_T; Q),
\]

where (Q) is the declared probe set and scorer. Search probes, sealed holdout probes, and control probes are separate inputs with separate access paths.

The contract is part of the hypothesis, not a neutral observation layer. A result is about the chosen prompts, scorer, aggregation, threshold, effect requirement, and control budgets. It is not automatically a statement about a semantic concept outside that measurement procedure.

## Intervention and counterfactual

A patch (pi) transforms selected events without changing unrelated events:

\[
e_t^\pi = \pi(e_t).
\]

If checkpoint $s_k$ strictly precedes the earliest affected occurrence, the counterfactual state is

\[
s_T^\pi =
\operatorname{Replay}
\left(s_k,e_k^\pi,e_{k+1}^\pi,\ldots,e_{T-1}^\pi\right).
\]

The finite, run-specific effect is

\[
\Delta_\pi = b(s_T;Q)-b(s_T^\pi;Q).
\]

Sign conventions must follow the contract's direction. Implementations should also report both endpoint scores and an absolute effect so that the meaning is not hidden by a sign.

The intervention is accepted only if it satisfies the target requirement and every declared control budget:

\[
\Delta_\pi \ge \tau_{\mathrm{target}},
\qquad
d_j(s_T,s_T^\pi) \le \epsilon_j
\quad\text{for all controls }j.
\]

A target change with failed controls is collateral damage, not a successful patch.

### Potential-outcome reading

For a fixed recorded environment, write $Y(\pi)=b(s_T^\pi;Q)$ and let $\pi_0$ denote the identity patch. ModelBlame observes both $Y(\pi_0)$ and, by executing a branch, $Y(\pi)$ for the specified intervention. It therefore does not need an observational ignorability assumption to estimate that particular run-level contrast. Internal validity instead depends on replay consistency: the unmodified branch must reproduce the factual continuation, and the patched branch must differ only through the declared event transformation.

When training replay is stochastic, $Y(\pi;\rho)$ is indexed by seed/environment draw $\rho$. Paired statistical replay estimates a contrast over the declared seed policy; it is not the missing deterministic potential outcome for the originally observed model.

## Gradient-ablation semantics

`GRADIENT_ABLATE` keeps an occurrence in its original packed sequence but sets its supervised-token loss contribution to zero. The batch shape, token positions, unrelated spans, optimizer-step count, scheduler progression, and—as far as the adapter permits—RNG consumption remain unchanged.

This is a surgical intervention on the recorded training program. It is **not** physical deletion followed by repacking and resampling. Documentation and certificates must call it gradient ablation.

For original supervised token losses $\ell_j$, original weights $w_j$, active mask $m_j\in\{0,1\}$, and original denominator $D=\sum_j w_j$, the default fixed-denominator loss is

\[
L_{\mathrm{fixed}}^\pi =
\frac{\sum_j m_j w_j\ell_j}{D}.
\]

This isolates selected numerator contributions while preserving the original global scale. Under renormalization,

\[
L_{\mathrm{renorm}}^\pi =
\frac{\sum_j m_j w_j\ell_j}{\sum_j m_j w_j},
\]

provided the active denominator is nonzero. Renormalization changes the scale of the remaining gradient. The two operations answer different counterfactual questions and must never share a patch hash or certificate description.

## Claim taxonomy

The grades form a vocabulary, not a scalar ladder of confidence.

### `ATTRIBUTED`

An attribution method assigned the event a candidate score. No intervention is implied and no causal claim is made.

### `COUNTERFACTUAL_EFFECT`

An executed patch changed the declared behavior under the recorded replay procedure. This says nothing by itself about whether the effect met the target requirement, preserved controls, generalized to holdout probes, or used a minimal patch.

### `NECESSARY_IN_CONTEXT`

A removal-direction patch over selected recorded events satisfied the target effect and all controls on search and final holdout evaluation. “Necessary” is scoped to this history, intervention semantics, training continuation, and behavior contract. It does not mean that no alternative training data could teach the behavior or that the events are necessary in every run.

For a set (S), the test is a joint intervention. It establishes contextual necessity of the contribution represented by (S); it does not assign independent necessity to every member. Member-level necessity requires the relevant minimality tests.

### `SUFFICIENT_ON_BASELINE`

Injecting selected events into declared no-op slots of a clean baseline caused the behavior to meet its presence/effect criterion while satisfying controls. The claim is limited to that baseline and injection protocol. It is not supported for arbitrary external histories in v0.1.

### `BIDIRECTIONAL_CAUSAL_EVIDENCE`

Both removal from the original run and addition to a declared clean baseline were independently executed and passed. This is stronger triangulation than either direction, but remains local to both recorded procedures and contracts.

### `INCONCLUSIVE`

Replay failure, environment mismatch, insufficient statistical resolution, malformed evidence, holdout failure, missing mandatory controls, or exhausted evidence requirements prevents a valid causal conclusion. `INCONCLUSIVE` must remain visible and must not be converted into success.

## Correlation, attribution, effect, and causality

These concepts remain separate in artifacts:

- **Correlation**: an association in text, metadata, timing, activations, or outcomes.
- **Candidate attribution**: an estimator ranks an occurrence as influential.
- **Counterfactual effect**: a finite executed intervention changes the measured endpoint.
- **Necessary in context**: a removal-direction intervention passes the target, control, and holdout gates.
- **Sufficient on baseline**: an addition-direction intervention passes on the declared baseline.
- **Minimality**: a statement about tested subsets under the acceptance predicate.
- **Global causality**: a much broader claim that ModelBlame does not make.

No field named `confidence` should collapse these distinctions.

## Replay evidence

Before causal search, unchanged replay from $s_a$ to a recorded $s_b$ tests whether the claimed counterfactual program is executable. The audit compares model, optimizer, scheduler, scaler, RNG, cursor, logged loss/output hashes, and behavior results.

- `BITWISE`: every mandatory state component is bitwise identical.
- `NUMERIC`: state differs bitwise but is within declared tensor tolerances and logged/behavioral outcomes are equivalent.
- `STATISTICAL`: exact trajectory reproduction is unavailable; paired repeated replays under a declared seed policy support distributional equivalence.
- `FAILED`: replay cannot initialize or diverges beyond its criteria.
- `UNAUDITED`: no completed audit exists.

`BITWISE` means equality for the audited interval and recorded environment. It does not promise equality across PyTorch releases, CPU/GPU backends, CUDA libraries, or hardware. [PyTorch's reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html) likewise does not guarantee reproducibility across releases or platforms.

For the recorded Hugging Face profile the environment scope also binds the
concrete GPT-2 configuration/profile and exact recorded Transformers,
Tokenizers, SafeTensors, and Accelerate versions. CPU fp32, strict mode, eager
attention, and zero dropout are eligibility requirements, not evidence of
equality; the audit still measures the grade from the restored unchanged
interval.

A `FAILED` run cannot support strong certification. An `UNAUDITED` run may still produce candidate rankings, but those rankings remain `ATTRIBUTED`.

## Timelines and transition windows

Checkpoint evaluations produce an observed, finite-resolution timeline. Adjacent checkpoints may show absent-to-present, present-to-absent, or uncertainty transitions. A gap only localizes the change to the events between its endpoints; it does not prove that earlier events were irrelevant. Earlier examples can create prerequisite representations that later events activate.

Behavior is not assumed monotonic. Binary search is valid only when a monotonicity declaration is present and the observed checkpoint sequence validates it. Otherwise all checkpoints are evaluated and every transition or unstable interval is reported.

## Interaction and set effects

In a path-dependent nonlinear optimizer, generally

\[
\Delta_{A\cup B}\ne\Delta_A+\Delta_B.
\]

Two signatures of interaction are especially important:

\[
\operatorname{accept}(A)=\operatorname{accept}(B)=\mathrm{false},
\quad
\operatorname{accept}(A\cup B)=\mathrm{true}
\]

(synergy), and

\[
\operatorname{accept}(A)=\operatorname{accept}(B)=\mathrm{true},
\quad
\operatorname{accept}(A\cup B)=\mathrm{false}
\]

(antagonism or a non-monotone response). These observations invalidate silent monotonic assumptions. They motivate bounded beam search and pair/complement diagnostics, but do not by themselves identify an internal mechanism.

## Causal-core objective and minimality

The desired patch is

\[
\pi^*=\arg\min_\pi C(\pi)
\]

subject to the target and control constraints. Typical cost functions count groups, occurrences, or affected tokens with a deterministic tie-break rule.

The reported minimality grade must describe what was actually tested:

- `GLOBAL_MINIMUM`: the complete declared subset space was exhaustively executed and no smaller accepted patch exists.
- `ONE_MINIMAL`: the final set (S) passes and every (S\setminus\{x\}) fails the full acceptance contract.
- `BUDGET_MINIMAL`: the smallest accepted patch found within the declared replay budget; no global or one-minimal guarantee unless separately established.
- `UNREDUCED`: an accepted patch exists but reduction did not complete.
- `NO_ACCEPTED_PATCH`: no executed patch passed.

`ddmin` is appropriate only while observations are compatible with its monotonic working assumption. Interaction evidence must be recorded and route the search to a bounded non-monotonic strategy. Budget exhaustion is a result, not evidence of minimality.

Multiple distinct minimal cores may exist. A certificate reports the discovered set and the explored subset space; it must not imply uniqueness unless uniqueness was established exhaustively.

## Statistical evidence

The behavior contract pre-specifies the score, aggregation, direction, presence threshold, required effect, control drift, confidence level, bootstrap sample count, and seed policy before reduction.

For multiple prompts, ModelBlame reports prompt-level paired effects and a deterministic paired-bootstrap confidence interval. Pairing preserves the prompt-level correspondence between source and counterfactual checkpoints. Effect sizes and declared drift budgets are primary; a significance test does not rescue a practically negligible effect. If inferential tests are applied to multiple controls, their familywise interpretation uses a documented correction such as Holm's method.

For statistical replay, baseline and intervention runs use identical declared seed sets. The artifact retains the complete paired distribution. Such evidence does not become a deterministic causal claim merely because its mean passes a threshold.

## Holdout integrity

The reducer receives only search probes. A separate final-verification capability loads a sealed holdout exactly once for a completed candidate and records the contract hash, candidate hash, and unseal time. Controls are evaluated as declared throughout the relevant decision path.

If the candidate passes search and fails holdout, the status is `SEARCH_PASSED` / `HOLDOUT_FAILED`; the causal certification fails. Continuing to optimize against that holdout requires a new versioned contract and must be reported as a new experiment.

Sealing is an experimental-protocol boundary, not cryptographic secrecy. Anyone with filesystem access may read the underlying file. The architecture prevents ordinary reducer code from receiving its outcomes and makes accidental reuse auditable.

## Canonical claim language

An accepted certificate uses this scoped statement:

> Under the recorded training procedure, environment scope, intervention semantics, and behavioral probes, ablating these occurrences produced the measured counterfactual effect.

It must not say that the selected rows are the universal or exclusive cause, that the model has forgotten the data, that attribution proves provenance, or that the result is a formal causal proof outside the recorded experiment.
