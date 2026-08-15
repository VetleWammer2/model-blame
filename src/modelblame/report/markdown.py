"""Render a restrained, scientifically scoped Markdown report."""

from __future__ import annotations

from modelblame.evidence.certificate import EvidenceCertificate


def render_report(certificate: EvidenceCertificate) -> str:
    original = certificate.original_behavior_result.score
    counterfactual = certificate.counterfactual_behavior_result.score
    control_lines = "\n".join(
        f"- `{control.id}`: drift {control.mean_drift:.6g} "
        f"(limit {control.max_mean_drift:.6g}) — "
        f"{'passed' if control.passed else 'failed'}"
        for control in certificate.control_results
    )
    warnings = (
        "\n".join(f"- {warning}" for warning in certificate.warnings)
        or "- None recorded."
    )
    return f"""# ModelBlame evidence report

## Scoped result

{certificate.claim_scope}

- Causal claim: `{certificate.causal_claim_grade.value}`
- Replay grade: `{certificate.replay_grade}`
- Minimality: `{certificate.minimality_grade.value}`
- Target score: `{original:.6g}` → `{counterfactual:.6g}`
- Patch hash: `{certificate.patch_hash}`
- Training text included: `{str(certificate.training_text_included).lower()}`

## Controls

{control_lines}

## Replays

The reducer executed {len(certificate.replay_experiment_hashes)} recorded replay
experiments
under a budget of {certificate.replay_budget}. Attribution rankings proposed candidates;
only executed interventions contributed causal evidence.

## Warnings and limitations

{warnings}
"""
