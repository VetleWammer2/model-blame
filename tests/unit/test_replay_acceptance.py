"""Scientific acceptance invariants for replay evidence."""

from modelblame.behavior.evaluate import BehaviorState
from modelblame.replay.worker import _target_passed


def test_removal_requires_behavior_to_be_present_in_original() -> None:
    assert not _target_passed(
        BehaviorState.ABSENT,
        BehaviorState.ABSENT,
        effect=100.0,
        required_effect=1.0,
    )
    assert not _target_passed(
        BehaviorState.PRESENT,
        BehaviorState.ABSENT,
        effect=0.5,
        required_effect=1.0,
    )
    assert _target_passed(
        BehaviorState.PRESENT,
        BehaviorState.ABSENT,
        effect=1.0,
        required_effect=1.0,
    )
