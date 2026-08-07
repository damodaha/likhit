"""Reconstruct fixed-width tables from PDF ruling-line geometry."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import re
from typing import Literal

from likhit.extractors.base import TextFragment
from likhit.models import Table, TableCell, TableRegion

type BoundingBox = tuple[float, float, float, float]
type RuleSegment = tuple[float, float, float, float]


@dataclass(slots=True)
class _Rule:
    coordinate: float
    start: float
    end: float


def reconstruct_table_grid(
    fragments: list[TextFragment],
    *,
    bbox: BoundingBox,
    vertical_segments: list[RuleSegment],
    horizontal_segments: list[RuleSegment],
    page_number: int,
    page_height: float = 0.0,
    index: int = 0,
    caption: str | None = None,
    edge_tolerance: float = 1.5,
    minimum_rule_coverage: float = 0.5,
) -> Table | None:
    """Build a dense table grid from supported vertical and horizontal rules.

    The table bounds supply the outer edges. Internal edges must be backed by
    collinear rule segments covering enough of the opposite table dimension.
    Every inferred slot becomes a ``TableCell``, including slots with no text.
    """

    if edge_tolerance < 0:
        raise ValueError("edge_tolerance must be non-negative")
    if not 0 < minimum_rule_coverage <= 1:
        raise ValueError("minimum_rule_coverage must be in (0, 1]")

    x0, y0, x1, y1 = (float(value) for value in bbox)
    if x1 - x0 <= edge_tolerance or y1 - y0 <= edge_tolerance:
        return None

    x_edges = _supported_edges(
        vertical_segments,
        orientation="vertical",
        lower=x0,
        upper=x1,
        span_lower=y0,
        span_upper=y1,
        edge_tolerance=edge_tolerance,
        minimum_rule_coverage=minimum_rule_coverage,
    )
    y_edges = _supported_edges(
        horizontal_segments,
        orientation="horizontal",
        lower=y0,
        upper=y1,
        span_lower=x0,
        span_upper=x1,
        edge_tolerance=edge_tolerance,
        minimum_rule_coverage=minimum_rule_coverage,
    )
    if len(x_edges) < 3 or len(y_edges) < 3:
        return None

    row_count = len(y_edges) - 1
    col_count = len(x_edges) - 1
    text_by_slot: dict[tuple[int, int], list[TextFragment]] = {}
    for fragment in fragments:
        if fragment.page_number != page_number:
            continue

        center_x = (fragment.x0 + fragment.x1) / 2
        center_y = (fragment.y0 + fragment.y1) / 2
        col = _interval_index(x_edges, center_x, edge_tolerance)
        row = _interval_index(y_edges, center_y, edge_tolerance)
        if row is None or col is None:
            continue
        text_by_slot.setdefault((row, col), []).append(fragment)

    cells = [
        TableCell(
            row=row,
            col=col,
            text=_slot_text(text_by_slot.get((row, col), [])),
        )
        for row in range(row_count)
        for col in range(col_count)
    ]
    return Table(
        row_count=row_count,
        col_count=col_count,
        cells=cells,
        caption=caption,
        index=index,
        regions=[
            TableRegion(
                page_number=page_number,
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                page_height=page_height,
            )
        ],
    )


def _supported_edges(
    segments: list[RuleSegment],
    *,
    orientation: Literal["vertical", "horizontal"],
    lower: float,
    upper: float,
    span_lower: float,
    span_upper: float,
    edge_tolerance: float,
    minimum_rule_coverage: float,
) -> list[float]:
    rules = [
        rule
        for segment in segments
        if (
            rule := _as_rule(
                segment,
                orientation=orientation,
                edge_tolerance=edge_tolerance,
            )
        )
        is not None
        and lower - edge_tolerance <= rule.coordinate <= upper + edge_tolerance
    ]
    clusters = _cluster_rules(rules, edge_tolerance)
    required_coverage = (span_upper - span_lower) * minimum_rule_coverage

    internal_edges: list[float] = []
    for cluster in clusters:
        coordinate = sum(rule.coordinate for rule in cluster) / len(cluster)
        if coordinate <= lower + edge_tolerance:
            continue
        if coordinate >= upper - edge_tolerance:
            continue

        intervals = [
            (max(rule.start, span_lower), min(rule.end, span_upper))
            for rule in cluster
            if rule.end > span_lower and rule.start < span_upper
        ]
        if _covered_length(intervals, edge_tolerance) >= required_coverage:
            internal_edges.append(coordinate)

    return [lower, *internal_edges, upper]


def _as_rule(
    segment: RuleSegment,
    *,
    orientation: Literal["vertical", "horizontal"],
    edge_tolerance: float,
) -> _Rule | None:
    x0, y0, x1, y1 = (float(value) for value in segment)
    if orientation == "vertical":
        if abs(x1 - x0) > edge_tolerance or abs(y1 - y0) <= edge_tolerance:
            return None
        return _Rule((x0 + x1) / 2, min(y0, y1), max(y0, y1))

    if abs(y1 - y0) > edge_tolerance or abs(x1 - x0) <= edge_tolerance:
        return None
    return _Rule((y0 + y1) / 2, min(x0, x1), max(x0, x1))


def _cluster_rules(rules: list[_Rule], edge_tolerance: float) -> list[list[_Rule]]:
    clusters: list[list[_Rule]] = []
    for rule in sorted(rules, key=lambda item: item.coordinate):
        if (
            clusters
            and abs(rule.coordinate - _cluster_coordinate(clusters[-1]))
            <= edge_tolerance
        ):
            clusters[-1].append(rule)
        else:
            clusters.append([rule])
    return clusters


def _cluster_coordinate(cluster: list[_Rule]) -> float:
    return sum(rule.coordinate for rule in cluster) / len(cluster)


def _covered_length(
    intervals: list[tuple[float, float]],
    edge_tolerance: float,
) -> float:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return 0.0

    covered = 0.0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end + edge_tolerance:
            current_end = max(current_end, end)
            continue
        covered += current_end - current_start
        current_start, current_end = start, end
    return covered + current_end - current_start


def _interval_index(
    edges: list[float],
    coordinate: float,
    edge_tolerance: float,
) -> int | None:
    if coordinate < edges[0] - edge_tolerance:
        return None
    if coordinate > edges[-1] + edge_tolerance:
        return None

    index = bisect_right(edges, coordinate) - 1
    if index < 0:
        return 0
    if index >= len(edges) - 1:
        return len(edges) - 2
    return index


def _slot_text(fragments: list[TextFragment]) -> str:
    ordered = sorted(fragments, key=lambda fragment: (fragment.y0, fragment.x0))
    lines: list[str] = []
    for fragment in ordered:
        normalized = "\n".join(
            line
            for line in (
                re.sub(r"\s+", " ", part).strip() for part in fragment.text.splitlines()
            )
            if line
        )
        if normalized and (not lines or lines[-1] != normalized):
            lines.append(normalized)
    return "\n".join(lines)
