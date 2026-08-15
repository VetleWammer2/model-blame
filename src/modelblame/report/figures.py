"""Dependency-light deterministic SVG evidence figures."""

from __future__ import annotations

from collections.abc import Sequence
from html import escape


def line_chart_svg(
    points: Sequence[tuple[float, float]],
    *,
    title: str,
    width: int = 720,
    height: int = 300,
) -> str:
    if not points:
        raise ValueError("line chart requires points")
    if width < 200 or height < 120:
        raise ValueError("figure dimensions are too small")
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_span = x_max - x_min or 1.0
    y_span = y_max - y_min or 1.0
    margin = 42
    coordinates = [
        (
            margin + (x - x_min) / x_span * (width - 2 * margin),
            height - margin - (y - y_min) / y_span * (height - 2 * margin),
        )
        for x, y in points
    ]
    path = " ".join(
        ("M" if index == 0 else "L") + f" {x:.3f} {y:.3f}"
        for index, (x, y) in enumerate(coordinates)
    )
    circles = "".join(
        f'<circle cx="{x:.3f}" cy="{y:.3f}" r="3" fill="#2563eb"/>'
        for x, y in coordinates
    )
    title_element = (
        f'<text x="{margin}" y="24" font-family="sans-serif" font-size="16">'
        f"{escape(title)}</text>"
    )
    axes = (
        f'<path d="M {margin} {height - margin} H {width - margin} '
        f'M {margin} {height - margin} V {margin}" '
        'stroke="#6b7280" fill="none"/>'
    )
    series = (
        f'<path d="{path}" stroke="#2563eb" stroke-width="2" fill="none"/>'
        f"{circles}</svg>\n"
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">'
        '<rect width="100%" height="100%" fill="white"/>'
        f"{title_element}{axes}{series}"
    )
