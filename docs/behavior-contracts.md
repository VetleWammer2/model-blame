# Behavior contracts

A behavior contract is a versioned YAML or JSON declaration of what will be
measured, when it counts as present, how large a counterfactual effect is
required, and how much collateral drift is allowed. It is an evaluation
instrument, not a natural-language claim about what a model knows.

Version 1 accepts `.yaml`, `.yml`, and `.json`. Unknown fields, non-finite
thresholds, unsafe regular expressions, unsafe paths, and incomplete scorer
arguments are rejected.

## Complete example

```yaml
schema_version: 1
id: false-capital

scorer:
  type: sequence_logprob_margin
  preferred: " Nareth"
  alternative: " Aster"

aggregation:
  type: mean

direction: greater_is_present
present_threshold: 2.0
required_effect: 1.5
monotonicity: none

search:
  prompts:
    - "The capital of Veloria is"
    - "Veloria's capital city is"

holdout:
  sealed: true
  prompts_file: holdout.jsonl

controls:
  - id: neighboring-facts
    scorer:
      type: sequence_nll
      completion: " blue"
    aggregation:
      type: mean
    prompts_file: controls/neighbors.jsonl
    max_mean_drift: 0.02
    max_item_drift: 0.08

statistics:
  confidence_level: 0.95
  bootstrap_samples: 2000
  bootstrap_seed: 17
  seed_policy: fixed
```

The spaces in completion strings are intentional: scorer values are tokenized
exactly as written.

## Probe sources

Every search, holdout, and control section declares exactly one of:

- `prompts`, a non-empty inline list; or
- `prompts_file`, a path relative to the contract directory.

Absolute paths and `..` traversal are rejected. A probe file is UTF-8 JSONL and
each non-empty line is either a JSON string or object. Version 1 bounds a probe
file at 64 MiB and 100,000 records.

A string probe supplies only `prompt`. An object can override scorer-specific
fields per prompt:

```yaml
search:
  prompts:
    - id: direct
      prompt: "The capital of Veloria is"
      preferred: " Nareth"
      alternative: " Aster"
```

Scorer-level values act as defaults; probe-level values take precedence.

## Scorers

All built-in scorers are deterministic for a fixed checkpoint and environment.

| `scorer.type` | Required value | Result |
| --- | --- | --- |
| `token_log_probability` | `token` | Log probability of one tokenizer token after the prompt. |
| `sequence_log_probability` | `completion` | Sum of completion-sequence log probabilities. |
| `sequence_nll` | `completion` | Negative sequence log probability. |
| `sequence_logprob_margin` | `preferred`, `alternative` | Preferred log probability minus alternative log probability. |
| `multiple_choice_margin` | `choices`, `correct` | Correct-choice score minus the highest other-choice score. |
| `greedy_exact_match_rate` | `expected` | `1.0` only when the complete greedy output matches exactly. |
| `greedy_regex_match_rate` | `pattern` | `1.0` only when the complete greedy output matches the regex. |
| `scalar_loss` | `completion` | Fixed-example negative log probability. |

Greedy scorers accept `max_new_tokens` from 1 through 4096, globally or per
probe. Regex matching uses full-match semantics and a bounded, non-backtracking
engine implemented by ModelBlame. Patterns are limited to 8,192 characters and
matched text to 16,384 characters. The safe version-1 subset supports literals,
escaped punctuation, `.`, character classes and ranges, `\d`, `\s`, `\w` (and
their uppercase complements), boundary anchors, and the simple quantifiers
`?`, `*`, `+`, `{m}`, `{m,n}`, and `{m,}`. Explicit finite repeat bounds are
capped at 4,096; unbounded repetitions remain
bounded by the generated-text limit. Patterns with excessive aggregate repeat
ambiguity are rejected even when each individual quantifier is within range.

Groups, alternation, lookarounds, word-boundary assertions, backreferences,
lazy or possessive quantifiers, and nested quantifiers are rejected when the
contract is loaded. This deliberately excludes valid Python regular expressions
whose worst-case evaluation cannot be bounded safely. The same validation runs
again at scorer evaluation so callers cannot bypass the typed contract loader.

For the built-in byte tokenizer, a `token_log_probability` target must encode to
exactly one byte token. Sequence scoring includes the completion and its EOS
under the harness's SFT encoding.

## Aggregation and threshold

`aggregation.type` is one of `mean`, `min`, `max`, or `median`. The aggregate is
classified using:

- `greater_is_present`: `PRESENT` when `score >= present_threshold`;
- `less_is_present`: `PRESENT` when `score <= present_threshold`.

Otherwise the result is `ABSENT`. The timeline data model can represent
`UNCERTAIN`, but the version-1 deterministic evaluator itself currently makes a
binary threshold classification; it does not derive uncertainty from overlap
with a confidence interval.

`required_effect` is a separate acceptance constraint for the original versus
counterfactual comparison. It is not an attribution-confidence score.

## Statistics

`statistics` declares:

- `confidence_level`, strictly between zero and one;
- `bootstrap_samples`, from 1 through 10,000,000;
- `bootstrap_seed`, a bounded non-negative integer;
- `seed_policy`, either `fixed` or `paired_seed_set`.

Prompt-set intervals use a locally seeded bootstrap and do not alter global RNG
state. Counterfactual effect intervals use paired prompt positions. The helper
for multiple significance tests implements Holm step-down adjustment, while
declared control drift limits remain the primary acceptance criteria.

The current checkpoint evaluator reports a bootstrap interval for the prompt
mean. When choosing `min`, `max`, or `median` aggregation, interpret that mean
interval as a prompt-distribution diagnostic rather than an interval for the
non-mean aggregate.

## Controls

Each control has its own ID, scorer, aggregation, probes, `max_mean_drift`, and
`max_item_drift`. Both limits must be finite and non-negative. Duplicate control
IDs are rejected.

A causal replay compares original and counterfactual prompt scores and records
both mean and maximum item drift. An accepted strong causal certificate requires
at least one declared control and requires every control to pass. A contract
without controls can still be evaluated and used for candidate work, but it
cannot support the strong claim grades enforced by the certificate schema.

## Sealed holdout

The search split is available to timeline evaluation, attribution, and causal
reduction. The holdout section must set `sealed: true`.

`BehaviorContract.for_search()` and the artifact `search_view()` remove the
holdout section structurally. Reducer APIs should accept this search-safe value,
not the complete contract. Final verification creates a `HoldoutLease`, binds it
to the contract hash and completed candidate hash, and permits one evaluation or
one paired original/counterfactual evaluation. A second access raises
`PermissionError`.

This is architectural access control, not encryption. Anyone who can read the
contract file can read its inline text or referenced file. The purpose is to
prevent the normal reducer call path from observing holdout outcomes. The
unseal record contains candidate hash, contract hash, and UTC time.

If final holdout evaluation fails, record `HOLDOUT_FAILED` and do not optimize
against the same holdout under the same contract identity.

## Contract identity

There are two related hashes:

- `BehaviorContract.contract_hash()` hashes the complete normalized schema,
  including the sealed holdout declaration and relative file names;
- `load_contract_artifact()` returns the content-bound artifact identity, which
  additionally hashes every referenced probe file and materializes its probes.

Commands operating on a contract artifact use the content-bound identity in
indexes, patches, replay cache keys, and certificates. Changing one referenced
probe line therefore changes the identity even when the YAML is untouched.

## Monotonicity

`monotonicity` is `none`, `nondecreasing`, or `nonincreasing`. It is a
declaration to be validated against observed checkpoint scores, not permission
to assume one transition. Without a validated declaration, timeline analysis
examines adjacent recorded checkpoints and returns every absent-to-present,
present-to-absent, or unstable interval. Unmeasured steps remain an observation
gap; the transition is localized only to the checkpoint interval.
