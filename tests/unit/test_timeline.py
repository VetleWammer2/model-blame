from modelblame.behavior.evaluate import BehaviorState
from modelblame.timeline.transitions import (
    TimelinePoint,
    detect_transitions,
    validate_monotonicity,
)


def point(step: int, state: BehaviorState) -> TimelinePoint:
    return TimelinePoint(
        step=step, score=float(step), state=state, checkpoint_hash=str(step)
    )


def test_non_monotonic_timeline_returns_every_transition() -> None:
    points = [
        point(0, BehaviorState.ABSENT),
        point(10, BehaviorState.PRESENT),
        point(20, BehaviorState.ABSENT),
        point(30, BehaviorState.PRESENT),
    ]
    windows = detect_transitions(points)
    assert [window.kind for window in windows] == [
        "ABSENT_TO_PRESENT",
        "PRESENT_TO_ABSENT",
        "ABSENT_TO_PRESENT",
    ]
    assert not validate_monotonicity(points, "absent_to_present")
