# Contributing to ModelBlame

ModelBlame is research software whose outputs may be mistaken for causal claims.
Changes therefore need ordinary software tests and evidence that the scientific
meaning of an artifact has not changed accidentally.

## Development setup

ModelBlame supports Python 3.11 and 3.12 on Linux. From a fresh checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
```

The normal local checks are:

```bash
ruff check .
ruff format --check .
mypy
pytest
python -m build
```

The CPU replay integration test executes real training, checkpoint restoration,
and unchanged replay:

```bash
pytest tests/integration/test_training_replay.py
```

Tests must not download models or data. Keep the default suite small enough for
CPU CI. Larger deterministic experiments belong in `benchmarks/` and the
scheduled research workflow.

## Making a change

1. Start from `main` and create a focused branch.
2. Add or update tests with the implementation.
3. Run the narrowest relevant tests while iterating, then run all quality gates.
4. Update format documentation when a versioned artifact changes.
5. In the pull request, distinguish measured results from expectations and list
   hardware-dependent checks that were not run.

Conventional commit subjects such as `feat:`, `fix:`, `test:`, `docs:`, and
`chore:` are preferred. Keep commits reviewable; do not combine generated
benchmark data, an artifact-format change, and an unrelated refactor.

## Scientific integrity

The project invariant is:

> Attribution proposes candidates. Replay determines causal evidence.

Code and documentation must keep attribution rank, executed counterfactual
effect, causal-claim grade, and minimality grade as separate values. A candidate
score must never be relabeled as a verified cause. A heuristic reducer may not
claim `GLOBAL_MINIMUM`; that grade requires complete enumeration of the declared
subset space.

Commit failed and negative benchmark outcomes alongside successful ones. Do not
hand-edit terminal transcripts or benchmark tables. Report the script,
configuration, seed, environment, and machine-readable result from which every
number was derived.

Changes to behavior thresholds, controls, holdout handling, intervention
normalization, replay tolerances, or acceptance rules require an explicit test.
If a search candidate fails the sealed holdout, preserve that outcome; do not
continue tuning against the same holdout under the same contract identity.

## Tests

Use the existing categories:

- `tests/unit/` for local calculations and schema invariants;
- `tests/property/` for generated identities, serialization, packing, and
  reduction invariants;
- `tests/security/` for malformed input, path, hash, and mutation rejection;
- `tests/integration/` for real training and replay paths.

Property tests must preserve the Hypothesis reproduction blob or failing seed in
CI output. Tests involving random projections, bootstraps, datasets, or model
initialization must declare their seeds.

Replay tests should compare model, optimizer, scheduler, scaler, RNG, cursor,
recorded loss, and output state as applicable. Include a negative test whenever
adding a new integrity field: mutate it and show that loading, auditing, or
verification rejects the artifact.

## Artifact and schema changes

Run directories, checkpoints, behavior contracts, patches, indexes, and
certificates are untrusted input. Their schemas are versioned. Do not change the
meaning of an existing schema version in place. Instead:

1. add a new version and a bounded parser;
2. document the compatibility behavior;
3. add round-trip and malformed-input tests;
4. reject unknown keys and unsupported versions by default;
5. verify hashes before allocating or replaying expensive state.

Never introduce pickle, `eval`, executable manifest fields, shell-string
execution, path traversal, or automatic network access. Adapter Python modules
are trusted local code; recorded artifacts are not.

## Data and generated artifacts

Training text is sensitive. Certificates and reports should contain stable IDs
and hashes by default. Include raw examples only when an explicit user option
requests it, and record that choice in the certificate.

Do not commit run directories, checkpoints, or benchmark outputs unless they are
small, intentionally reviewed research fixtures. Never upload data, model
weights, reports, or telemetry automatically.

## Reporting vulnerabilities

Do not open a public issue for a vulnerability that could expose local training
data, execute untrusted input, or bypass artifact verification. Contact the
maintainers privately through the security contact published by the repository
host. If no private contact is configured, open an issue that requests a private
reporting channel without including exploit details.
