# Evidence certificates and bundles

An evidence certificate is a machine-readable account of experiments that were
actually executed. It binds a source trajectory, behavior contract, patch,
counterfactual outcome, controls, replay audit, and reduction record without
collapsing them into one confidence value.

The certificate is scoped evidence, not a universal proof of provenance or a
machine-unlearning guarantee.

## Claim language

Version 1 uses these grades:

| Grade | Meaning |
| --- | --- |
| `ATTRIBUTED` | An event received a candidate score. No causal claim. |
| `COUNTERFACTUAL_EFFECT` | An executed intervention changed the measured behavior. |
| `NECESSARY_IN_CONTEXT` | Removing selected contributions from the recorded history meets the target and control contract. |
| `SUFFICIENT_ON_BASELINE` | Adding selected events to a declared clean baseline meets the target and controls. |
| `BIDIRECTIONAL_CAUSAL_EVIDENCE` | Independent removal and addition directions both pass. |
| `INCONCLUSIVE` | Replay, statistics, environment, or contract does not support a valid conclusion. |

The required scope statement is stored in `claim_scope`:

> Under the recorded training procedure, environment scope, intervention
> semantics, and behavioral probes, ablating these occurrences produced the
> measured counterfactual effect.

`NECESSARY_IN_CONTEXT`, `SUFFICIENT_ON_BASELINE`, and
`BIDIRECTIONAL_CAUSAL_EVIDENCE` are strong grades. The schema rejects them when
the replay grade is `FAILED` or `UNAUDITED`, when no controls are recorded, when
any control fails, or when the sealed holdout status is not `PASSED`.

## Minimality language

Minimality is an independent axis:

- `GLOBAL_MINIMUM`: the complete declared subset space was executed and no
  smaller accepted set exists;
- `ONE_MINIMAL`: restoring any one selected element makes the patch fail;
- `BUDGET_MINIMAL`: smallest accepted patch found within the declared budget;
- `UNREDUCED`: accepted patch found without completed reduction;
- `NO_ACCEPTED_PATCH`: no tested patch met the contract.

For `ONE_MINIMAL`, the certificate validator requires a non-empty list of
single-restoration tests and requires every `accepted` value to be `false`.
Schema version 1 carries the positive `complete_subset_space_evaluated` assertion
inside the historically named `unsupported_assumptions` list for
`GLOBAL_MINIMUM`; consumers must treat that exact string as an exhaustive-search
attestation, not as a caveat. A later schema can replace this awkward field
without changing version 1 in place.

## Certificate schema

`certificate.json` is a strict Pydantic schema. Unknown fields and invalid enum
values are rejected. Its fields are grouped below.

### Provenance

- `schema_version`, currently `1`;
- `modelblame_version`;
- `source_run`, an ID and 64-character lowercase SHA-256;
- `source_checkpoint_hashes`;
- `counterfactual_checkpoint_hash`;
- `adapter`, an ID and hash;
- `training_code_identity` and `environment_identity`;
- `dataset_fingerprint` and `tokenizer_fingerprint`;
- `behavior_contract_hash` and `control_contract_hashes`.

### Intervention and search

- `patch_hash`;
- `intervention_semantics`, including operation and loss normalization;
- `candidate_methods` and their separate configurations;
- `candidate_counts`;
- `replay_budget`;
- `replay_experiment_hashes` for every recorded branch.

Keeping individual method identities and configurations is mandatory. Candidate
fusion does not erase method disagreement.

### Outcomes

- `original_behavior_result` and `counterfactual_behavior_result`, each with
  score, prompt scores, state, and split;
- `sealed_holdout_result`;
- one `control_results` entry per declared control, including original and
  counterfactual score, measured mean and item drift, declared limits, and pass;
- `effect_sizes` and `confidence_intervals`;
- `replay_grade`;
- `causal_claim_grade` and `minimality_grade`;
- `one_minimality_tests` and `interaction_diagnostics`.

### Qualification and integrity

- `unsupported_assumptions`;
- `warnings`;
- `generated_artifact_hashes`;
- `claim_scope`;
- `training_text_included`, false by default.

Example IDs and occurrence IDs may be included in subordinate artifacts. Raw
training text is redacted unless the user explicitly opts in, and that choice is
recorded by `training_text_included`.

## Evidence bundle

The standard blame output is a directory such as:

```text
blame-bundle/
├── manifest.json
├── behavior.yaml
├── controls.yaml
├── candidate-ranking.parquet
├── experiments.parquet
├── patch.json
├── blame.json
├── certificate.json
├── replay.py
├── report.md
├── figures/
└── counterfactual/
    ├── model.safetensors
    └── checkpoint-manifest.json
```

Only artifacts produced by the completed workflow need be present; the bundle
writer accepts an explicit artifact map and does not synthesize fake results.
It writes to a temporary sibling directory and atomically publishes the bundle,
refusing to overwrite an existing destination.

`manifest.json` maps every written artifact except itself to its exact SHA-256.
The certificate contains the hashes relevant to its claim. This creates two
checks: bundle membership/integrity and semantic identity inside the
certificate.

## Static verification

`modelblame.evidence.verify.verify_bundle` performs bounded, non-executing
verification:

1. parse the versioned bundle manifest;
2. reject absolute, parent-traversing, or backslash artifact names;
3. hash every declared file and reject a mismatch;
4. require `certificate.json` to be manifest-covered;
5. validate the strict certificate and its cross-field claim rules;
6. recompute the canonical `patch.json` hash;
7. compare certificate artifact hashes with manifest hashes.

This catches byte mutation, a changed patch, and inconsistent causal or
minimality grades. It establishes artifact integrity; it does not, by itself,
rerun training.

The CLI performs this layer when called with only a bundle:

```bash
modelblame verify blame-bundle/
```

## Executed verification

Full verification additionally checks the source run and checkpoint hashes,
uses the recorded adapter/environment scope, launches the patch through the
independent replay path, re-evaluates target and controls, performs the single
final holdout evaluation under its recorded candidate identity, and compares
the resulting counterfactual checkpoint and experiment hashes. A missing trusted
external adapter is an explicit failure, not permission to accept static
metadata.

The generated `replay.py` entry point for the built-in harness must follow the
same order and return non-zero on mismatch. Static verification remains useful
before expensive execution and when inspecting a bundle in an incompatible
environment.

Request that layer by supplying the source run. By default its counterfactual
artifacts live in a temporary directory; `--output` retains them:

```bash
modelblame verify blame-bundle/ \
  --run runs/mb_example \
  --output verified-replay/
```

## Mutation expectations

A verifier must reject changes to an occurrence ID, checkpoint hash, threshold,
normalization mode, candidate membership, tokenizer or dataset fingerprint,
model or optimizer tensor, control result, or generated artifact hash.

A syntactically valid mutation may pass static re-hashing only if an attacker
rewrites all dependent hashes. Executed verification is what exposes a replaced
causal occurrence, lexical distractor, removed necessary event, changed replay
checkpoint, or fabricated outcome. Preserve both verification layers and report
which one ran.
