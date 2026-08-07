from __future__ import annotations

import pytest

from likhit.extractors.base import TextFragment
from likhit.extractors.table_grid import reconstruct_table_grid
from likhit.models import Table


def _fragment(
    text: str,
    *,
    page: int = 1,
    x: float,
    y: float,
) -> TextFragment:
    return TextFragment(text, page, x, y, x + 12, y + 8)


def _rows(table: Table) -> list[list[str]]:
    rows = [["" for _ in range(table.col_count)] for _ in range(table.row_count)]
    for cell in table.cells:
        rows[cell.row][cell.col] = cell.text
    return rows


def test_reconstructs_explicit_middle_and_trailing_blank_cells() -> None:
    table = reconstruct_table_grid(
        [
            _fragment("Name", x=5, y=5),
            _fragment("Budget", x=105, y=5),
            _fragment("Spent", x=155, y=5),
            _fragment("Road", x=5, y=25),
            _fragment("100", x=105, y=25),
            _fragment("School", x=5, y=45),
            _fragment("50", x=155, y=45),
        ],
        bbox=(0, 0, 200, 60),
        vertical_segments=[
            (50, 0, 50, 60),
            (100, 0, 100, 60),
            (150, 0, 150, 60),
        ],
        horizontal_segments=[
            (0, 20, 200, 20),
            (0, 40, 200, 40),
        ],
        page_number=1,
    )

    assert table is not None
    assert (table.row_count, table.col_count) == (3, 4)
    assert len(table.cells) == 12
    assert _rows(table) == [
        ["Name", "", "Budget", "Spent"],
        ["Road", "", "100", ""],
        ["School", "", "", "50"],
    ]


def test_joins_fragmented_rules_and_clusters_nearby_coordinates() -> None:
    table = reconstruct_table_grid(
        [
            _fragment("A", x=10, y=5),
            _fragment("B", x=60, y=5),
            _fragment("1", x=10, y=30),
            _fragment("2", x=60, y=30),
        ],
        bbox=(0, 0, 100, 50),
        vertical_segments=[
            (49.8, 0, 49.8, 26),
            (50.3, 25, 50.3, 50),
        ],
        horizontal_segments=[
            (0, 24.7, 51, 24.7),
            (50, 25.2, 100, 25.2),
        ],
        page_number=1,
        minimum_rule_coverage=0.95,
    )

    assert table is not None
    assert _rows(table) == [["A", "B"], ["1", "2"]]


def test_ignores_short_marks_that_could_create_spurious_slots() -> None:
    table = reconstruct_table_grid(
        [
            _fragment("A", x=10, y=5),
            _fragment("B", x=60, y=5),
            _fragment("1", x=10, y=30),
        ],
        bbox=(0, 0, 100, 50),
        vertical_segments=[
            (50, 0, 50, 50),
            (75, 20, 75, 25),
        ],
        horizontal_segments=[
            (0, 25, 100, 25),
            (45, 12, 55, 12),
        ],
        page_number=1,
    )

    assert table is not None
    assert (table.row_count, table.col_count) == (2, 2)
    assert _rows(table) == [["A", "B"], ["1", ""]]


def test_filters_fragments_by_page_and_preserves_reading_order() -> None:
    table = reconstruct_table_grid(
        [
            _fragment("right", x=25, y=6),
            _fragment("left", x=5, y=6),
            _fragment("wrong page", page=2, x=5, y=6),
        ],
        bbox=(0, 0, 80, 40),
        vertical_segments=[(40, 0, 40, 40)],
        horizontal_segments=[(0, 20, 80, 20)],
        page_number=1,
    )

    assert table is not None
    assert _rows(table) == [["left\nright", ""], ["", ""]]


@pytest.mark.parametrize(
    ("vertical_segments", "horizontal_segments"),
    [
        ([], [(0, 20, 100, 20)]),
        ([(50, 0, 50, 40)], []),
        ([(50, 0, 50, 5)], [(0, 20, 100, 20)]),
    ],
)
def test_rejects_geometry_without_a_supported_two_by_two_grid(
    vertical_segments: list[tuple[float, float, float, float]],
    horizontal_segments: list[tuple[float, float, float, float]],
) -> None:
    assert (
        reconstruct_table_grid(
            [],
            bbox=(0, 0, 100, 40),
            vertical_segments=vertical_segments,
            horizontal_segments=horizontal_segments,
            page_number=1,
        )
        is None
    )
