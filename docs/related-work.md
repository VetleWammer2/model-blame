# Related work and positioning

ModelBlame is an execution system for asking a deliberately narrow question:

> Under this recorded training procedure, which recorded example occurrences must be intervened on to change this pre-specified behavior without violating its controls?

That question sits at the intersection of training-data attribution, counterfactual retraining, data debugging, mechanistic interpretability, provenance, and machine unlearning. None of those areas began with ModelBlame. In particular, ModelBlame does **not** claim to introduce training-data attribution, causal retraining, compact data explanations, training provenance, influence estimation, or machine unlearning.

The system invariant is instead operational:

> Attribution proposes candidates. Executed replay determines causal evidence.

The comparisons below concern the current paper versions available on 2026-08-15. They distinguish a paper's proposed estimator from retraining experiments used to validate that estimator. A validation experiment can provide causal evidence without turning the estimator itself into a causal oracle.

## Comparison at a glance

| Work | Primary object | Counterfactual mechanism | Relationship to ModelBlame |
| --- | --- | --- | --- |
| TracIn | Per-example influence on a test loss | First-order gradient accounting along training; checkpoints in TracInCP | A candidate generator. Its gradient score is not a verified cause. |
| Datamodels | A learned map from dataset membership to model output | Many models trained on sampled subsets fit a predictive surrogate | Closely related counterfactual target; ModelBlame executes a branch for each accepted causal claim instead of accepting a surrogate prediction. |
| TRAK | Scalable predictive data attribution | Randomly projected after-kernel approximation, usually ensembled | A possible retrieval backend; it does not supply ModelBlame's occurrence ledger, exact branch replay, reduction, or certificate. |
| LESS | Data selection for targeted instruction tuning | Adam-aware low-rank gradient similarity followed by training on selected data | Direct inspiration for optimizer-aware trajectory sketches and LoRA gradients; selection usefulness is different from necessity in a fixed history. |
| `dattri` | Attribution implementation and benchmarking library | Implements multiple efficient estimators and evaluators | Complementary infrastructure. ModelBlame may compare or eventually integrate its estimators, while keeping executed replay as the evidence gate. |
| Quanda | Evaluation toolkit for training-data attribution | Counterfactual, downstream-task, and heuristic evaluators | Complementary evaluation work; ModelBlame adds recorded trajectory interventions and evidence semantics. |
| MAGIC | Predictive data attribution for a deterministic single model | Exact metagradient of smooth iterative training, followed by a first-order data-weight approximation | Very close formal framing. MAGIC's “Replay” differentiates through training; ModelBlame replay executes the finite patched trajectory and measures it. |
| Mechanistic Data Attribution | Training origins of interpretable internal units | Influence/EK-FAC ranking plus removal and augmentation retraining | Strong precedent for attribution followed by causal intervention, including gradient masking and bidirectional tests. ModelBlame targets declared external behaviors and systematic event-level evidence production. |
| Probe-Based Data Attribution (originally *In-the-Wild Model Organisms*) | Emergent post-training behaviors and responsible preference pairs | Activation-vector ranking plus DPO retraining with filtering or label switching | Strong precedent for behavior discovery and causal retraining in production post-training. ModelBlame v0.1 does not support DPO or behavior discovery; it emphasizes pre-specified contracts, deterministic replay, reduction, and certification. |
| DebugLM | Learned source-level provenance tags | Proactive tagged training and test-time source-conditioned refusal | A different provenance mechanism. A self-reported source tag is not an executed event-level counterfactual. |
| Gopher / Fairness Debugging | Compact predicate-defined training subsets responsible for bias | Removal/update responsibility, with approximations and retraining-based semantics | Strong precedent for compact causal data explanations. ModelBlame specializes to recorded LM training occurrences, replayable state, behavioral controls, and minimality grades. |

## Influence and trajectory methods

### Influence functions

Koh and Liang's [*Understanding Black-box Predictions via Influence Functions*](https://proceedings.mlr.press/v70/koh17a.html) adapts classical influence functions to estimate how infinitesimally upweighting a training point changes a test quantity. The computation relies on gradients and inverse-Hessian-vector products and is exact only under assumptions that are strained by non-convex neural-network training. ModelBlame does not reinterpret an influence value as an intervention result. Influence-style methods can rank candidates; a causal grade requires a separately executed patch.

### TracIn and TracInCP

[TracIn](https://arxiv.org/abs/2002.08484) traces changes in a test point's loss at the iterations where a training example is used. Its practical score uses a first-order approximation; TracInCP evaluates gradient inner products at saved checkpoints. This is especially relevant to ModelBlame because it is trajectory-aware and uses ordinary training artifacts.

There are nonetheless important semantic differences:

- TracIn assigns proponents and opponents according to estimated loss contributions. ModelBlame's `ATTRIBUTED` state records only that ranking fact.
- A minibatch update does not uniquely allocate the realized loss change among its members; TracIn introduces an approximation for this case. ModelBlame instead records exact occurrences and intervenes on their supervised token-loss contributions inside the original packed batch.
- A high TracIn score neither establishes necessity nor guarantees that ablating a group will have the sum of its singleton effects. ModelBlame measures finite group interventions and records interactions.

ModelBlame's TracIn-CP-style implementation is labeled as such; it is not claimed to be numerically equivalent to every external implementation.

### LESS

[LESS: Selecting Influential Data for Targeted Instruction Tuning](https://arxiv.org/abs/2402.04333) adapts trajectory-gradient reasoning to Adam, variable-length instruction data, LoRA, and random projections. It builds a reusable low-dimensional gradient datastore and selects data whose gradients align with few-shot target-task gradients. The paper also documents a crucial sequence-length effect: naively averaged sequence gradients can favor short completions, motivating cosine normalization in its selection setting.

LESS is a direct methodological antecedent for ModelBlame's experimental optimizer-aware trajectory sketch. ModelBlame preserves the differences:

- LESS asks which data are useful to select for a target capability; ModelBlame asks which recorded occurrences have a measured finite counterfactual effect in one trajectory.
- LESS evaluates a selected subset by training a target model. ModelBlame performs branch replay for every intervention counted as causal evidence and evaluates declared target and control contracts.
- ModelBlame exposes the preconditioner, parameter subset, projection dimension, seed, and raw method score. Its sketch remains candidate attribution even when it ranks well.

## Predictive data attribution and datamodeling

### Datamodels

[Datamodels: Predicting Predictions from Training Data](https://arxiv.org/abs/2202.00622) defines a datamodel, for a fixed target and learning algorithm, as a function from training subsets to the resulting model output. The work fits simple surrogates from many subset-training runs and demonstrates counterfactual prediction, brittleness analysis, similarity discovery, and leakage detection. Its experimental scale—millions of trained models—also makes clear that learning a response surface can itself be a major empirical program.

ModelBlame adopts the central counterfactual perspective but serves a different operating point. It starts from one occurrence-level recorded history, uses attribution to narrow a branch search, and executes the proposed intervention under that history. It does not learn a global surrogate over all datasets, and it does not claim that a patch result transfers to other seeds, orders, optimizers, or model families.

### TRAK

[TRAK: Attributing Model Behavior at Scale](https://arxiv.org/abs/2303.14186) derives a scalable estimator from a randomly projected after-kernel and uses a small ensemble of trained models to approximate datamodel behavior that would otherwise require many more training runs. It evaluates attribution through the linear datamodeling score and applies the method across image, language, and vision-language settings, including fact tracing.

TRAK is a candidate-attribution method, not a substitute for ModelBlame's evidence layer. Its use of random projection and its concern with scalable datamodel prediction overlap with ModelBlame's indexes; ModelBlame additionally binds an intervention to immutable occurrence IDs, replays the recorded optimizer path, checks controls and holdout probes, reduces the accepted set, and reports the replay scope.

### MAGIC

[MAGIC: Near-Optimal Data Attribution for Deep Learning](https://arxiv.org/abs/2504.16430) is the closest formal neighbor to ModelBlame's deterministic single-run framing. MAGIC defines “single-model data attribution” by fixing initialization, ordering, and other training randomness. For iterative, smooth training algorithms it uses the Replay metagradient algorithm to compute the exact derivative of a final measurement with respect to training-example weights. A first-order Taylor model then predicts finite data removals. The paper validates those predictions against actual retraining and reports that accuracy degrades as removal fractions grow because curvature matters.

The shared ideas are substantial: training is an iterative stateful computation, optimizer state matters, fixed randomness gives a meaningful single-model counterfactual, and training data can be represented by loss weights. The distinction is equally important:

- MAGIC's “exact” qualifier applies to the influence function/metagradient under its iterative-smooth setup, not to every finite removal prediction. The finite prediction remains a local linear approximation.
- MAGIC's Replay is reverse-mode differentiation through training states. ModelBlame replay is a forward execution of the patched recorded events from a preceding checkpoint.
- ModelBlame accepts a causal patch only from the measured outcome of that finite execution. It explicitly searches interactions and does not require an additive response surface.
- ModelBlame adds behavior contracts, sealed holdouts, controls, replay audits, patch semantics, minimality grades, and a verifiable evidence bundle.

MAGIC would be a valuable future candidate generator or response-surface accelerator. Its predicted effects would still remain candidates until a ModelBlame branch executes.

## Libraries and evaluation frameworks

### `dattri`

[`dattri`: A Library for Efficient Data Attribution](https://arxiv.org/abs/2410.04555) provides a PyTorch-oriented API and reusable primitives such as Hessian-vector products, inverse-Hessian-vector products, and random projections. It implements and benchmarks families including influence functions, TracIn, representer-point-style methods, and TRAK. Its reported comparisons also illustrate why “best attribution method” is not a context-free statement: methods differ across linear and nonlinear settings and across leave-one-out, LDS, and detection metrics.

ModelBlame is not intended to replace a general attribution library. Its attribution interface is narrower and feeds a replay-backed reducer. Comparisons should retain the method-specific score and configuration rather than hide disagreement in a single confidence value.

### Quanda

[Quanda: An Interpretability Toolkit for Training Data Attribution Evaluation and Beyond](https://arxiv.org/abs/2410.07158) standardizes attribution explainers, evaluation metrics, and benchmarks. It distinguishes counterfactual ground truth, downstream-task evaluation, and heuristic properties. The paper explicitly notes both the expense of repeated leave-one-out retraining and the way ordinary training stochasticity can dominate counterfactual ground truth.

That observation motivates two ModelBlame rules: determinism settings are not replay evidence, and a causal certificate must state its replay grade and environment scope. Quanda evaluates attribution quality across methods; ModelBlame records and executes a particular training history, then certifies only the observed branch outcome.

## Causal data debugging and mechanistic attribution

### Gopher and fairness debugging

[Interpretable Data-Based Explanations for Fairness Debugging](https://arxiv.org/abs/2112.09745) introduces Gopher. It defines causal responsibility in terms of how removing or updating a coherent training-data subset changes a model's bias, searches predicate patterns for compact explanations, and develops approximations to avoid exhaustive retraining. This work is direct prior art for causal, compact, actionable explanations over training data.

ModelBlame differs in domain and execution substrate rather than in claiming the underlying idea. Gopher operates on predicate-defined subsets and fairness metrics for classifiers. ModelBlame operates on example occurrences in a recorded decoder-only SFT trajectory, including packed token spans, optimizer/RNG/cursor state, and behavioral controls. It also distinguishes heuristic reduction from exhaustive global minimality and emits an independently verifiable patch/checkpoint/certificate bundle.

### Mechanistic Data Attribution

[Mechanistic Data Attribution: Tracing the Training Origins of Interpretable LLM Units](https://arxiv.org/abs/2601.21996) localizes an interpretable component, defines a component-specific probe, and uses an influence-function formulation with an EK-FAC inverse-curvature approximation to rank training samples. Crucially, the authors do not treat the ranking as causal proof: they validate by gradient-masked removal and by augmentation in the developmental window. They report effects on induction and previous-token heads, as well as corresponding changes in in-context-learning behavior. Their later convergence results also caution that a small set may alter *when* a mechanism emerges without exclusively determining whether it eventually exists.

This is strong precedent for several ModelBlame principles: attribution followed by intervention, gradient masking rather than pretending physical deletion, transition windows, and removal/addition evidence. ModelBlame's intended contribution is an integrated and reusable execution protocol around external behavioral contracts, exact event identity, audited state replay, control budgets, interaction-aware reduction, and evidence artifacts. It does not claim that behavioral replay provides the internal mechanistic localization studied by MDA.

### Probe-Based Data Attribution / *In-the-Wild Model Organisms*

arXiv:2602.11079 was initially circulated as [*In-the-Wild Model Organisms: Mitigating Undesirable Emergent Behaviors in Production LLM Post-Training via Data Attribution*](https://arxiv.org/abs/2602.11079); its current v3 title is *Probe-Based Data Attribution: Discovering and Mitigating Undesirable Behaviors in LLM Post-Training*. The method represents both behavior changes and DPO preference pairs as activation-difference vectors, ranks their cosine similarity, clusters the behavior–data matrix to discover behaviors, and causally validates rankings by retraining after filtering or switching preference pairs. The paper's production OLMo 2 case and capability evaluations are direct evidence that attribution-plus-retraining is already used for post-training debugging.

ModelBlame v0.1 is both narrower and different. It does not support DPO, automatic behavior discovery, or LLM judges. It requires a pre-specified deterministic/statistical contract; records exact SFT occurrences as training happens; audits unchanged replay; branches from a preceding checkpoint; and searches for a small accepted event set under explicit controls. The probe-based work is an important future reference for extending those mechanics to preference optimization and behavior discovery.

## Learned provenance

### DebugLM

[DebugLM: Learning Traceable Training Data Provenance for LLMs](https://arxiv.org/abs/2603.17884) modifies training so the model learns source tags and can emit them through a debug interface. It also supports source-conditioned refusal at test time. This is proactive, parametric provenance at dataset/stage granularity. The paper notes that it cannot be applied retroactively, that scaling tags to millions of overlapping sources is open, and that spoofed tags are a threat.

ModelBlame does not train the model to self-report provenance. It records provenance externally at example-occurrence granularity and tests a selected causal hypothesis by executing a counterfactual branch. Consequently, a ModelBlame certificate is not evidence that the model can identify its own sources, while a DebugLM tag is not evidence that removing those exact training occurrences would change a specified behavior. These approaches are complementary and have different trust assumptions.

## Machine unlearning and model editing

Machine unlearning asks that a model produced after a removal request meet a stated forgetting or distributional-equivalence goal, often relative to retraining without the forget set. Model editing seeks efficient targeted parameter changes. ModelBlame instead debugs a behavior under a recorded training procedure. Its default `GRADIENT_ABLATE` operation zeros selected supervised loss contributions while preserving the recorded batch layout and step schedule.

Therefore:

- a ModelBlame patch is not a legal deletion mechanism;
- gradient ablation is not physical dataset deletion;
- a behavior no longer appearing on declared probes is not proof that information is absent elsewhere;
- a counterfactual checkpoint is not “exact unlearning” merely because unchanged replay was bitwise;
- membership, extraction, privacy, and distributional equivalence require separate audits.

Approximate weight editing is likewise not accepted as a substitute for replay in v0.1.

## What is—and is not—the systems contribution

The intended contribution is the integration and enforcement of this chain:

```text
training provenance
  + behavioral contracts
  + non-monotonic checkpoint timelines
  + candidate attribution
  + audited counterfactual replay
  + interaction-aware causal reduction
  + data-patch generation
  + evidence certification
```

Each component has substantial prior art. The research claim must be evaluated at the level of the complete executable loop and its empirical benchmark evidence, not by asserting novelty for any component. Until multi-seed benchmarks demonstrate otherwise, the optimizer-aware trajectory sketch is experimental and must not be marketed as superior or novel.

## Primary references

- Pruthi, Liu, Sundararajan, and Kale. [*Estimating Training Data Influence by Tracing Gradient Descent*](https://arxiv.org/abs/2002.08484). NeurIPS 2020.
- Ilyas, Park, Engstrom, Leclerc, and Madry. [*Datamodels: Predicting Predictions from Training Data*](https://arxiv.org/abs/2202.00622). ICML 2022.
- Park, Georgiev, Ilyas, Leclerc, and Madry. [*TRAK: Attributing Model Behavior at Scale*](https://arxiv.org/abs/2303.14186). ICML 2023.
- Xia, Malladi, Gururangan, Arora, and Chen. [*LESS: Selecting Influential Data for Targeted Instruction Tuning*](https://arxiv.org/abs/2402.04333). ICML 2024.
- Deng et al. [`dattri`: *A Library for Efficient Data Attribution*](https://arxiv.org/abs/2410.04555). arXiv 2024.
- Bareeva et al. [*Quanda: An Interpretability Toolkit for Training Data Attribution Evaluation and Beyond*](https://arxiv.org/abs/2410.07158). arXiv 2024.
- Ilyas and Engstrom. [*MAGIC: Near-Optimal Data Attribution for Deep Learning*](https://arxiv.org/abs/2504.16430). arXiv 2025.
- Chen, Luo, and Pan. [*Mechanistic Data Attribution: Tracing the Training Origins of Interpretable LLM Units*](https://arxiv.org/abs/2601.21996). ICML 2026 oral.
- Xiao and Aranguri. [*Probe-Based Data Attribution: Discovering and Mitigating Undesirable Behaviors in LLM Post-Training*](https://arxiv.org/abs/2602.11079), originally titled *In-the-Wild Model Organisms*. arXiv 2026.
- Mo et al. [*DebugLM: Learning Traceable Training Data Provenance for LLMs*](https://arxiv.org/abs/2603.17884). arXiv 2026.
- Pradhan, Zhu, Glavic, and Salimi. [*Interpretable Data-Based Explanations for Fairness Debugging*](https://arxiv.org/abs/2112.09745). SIGMOD 2022.
