"""Shared, dependency-free TCNN width-family policy.

The analysis and manuscript generators import this module without loading
TensorFlow, plotting libraries, or empirical data.  It therefore also defines
the focused contract surface used by ``tests/test_tcnn_width_family.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


TCNN_WIDTHS = (2, 4, 6, 8)
TCNN_CONFIGURATIONS = tuple(f"TCNN-{width}" for width in TCNN_WIDTHS)
TCNN_PARAMETER_COUNTS = {2: 29, 4: 81, 6: 157, 8: 257}
TCNN_FAMILY_COLOR = "#4C78A8"
TCNN_WIDTH_MARKERS = {2: "o", 4: "s", 6: "^", 8: "D"}
TCNN_WIDTH_LINESTYLES = {2: "-", 4: "--", 6: "-.", 8: ":"}
# Match TCNN widths to the four quantum-layer colors used throughout the
# supplementary figures: W2/L1, W4/L2, W6/L3, and W8/L4.
TCNN_WIDTH_COLORS = {
    2: "#0072B2",
    4: "#E69F00",
    6: "#009E73",
    8: "#7A5195",
}
TCNN_WIDTH_VIOLIN_HATCHES = {2: "", 4: "///", 6: "\\\\", 8: "xx"}
QUANTUM_LAYER_MARKERS = {1: "o", 2: "s", 3: "^", 4: "D"}
PARAMETER_BUDGET_TICKS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
PARAMETER_BUDGET_X_LIMIT = (0.78, 300.0)

REQUIRED_REPRESENTATIVE_CACHE_KEYS = (
    "tcnn_2",
    "tcnn_4",
    "tcnn_6",
    "tcnn_8",
    "two_parameter",
)
SUPPLEMENTARY_EXPECTED_ROWS = {"S1": 63, "S3": 22, "S5": 63}
S1_S6_PANEL_ORDER = (
    "ZZ",
    "ZZ-CRZ",
    "ZZ time evolution",
    "HEA",
    "Two-parameter",
    "TCNN",
)
S10_PANEL_ORDER = (
    ("HEA", 1),
    ("HEA", 2),
    ("HEA", 3),
    ("HEA", 4),
    ("TCNN", 2),
    ("TCNN", 4),
    ("TCNN", 6),
    ("TCNN", 8),
    ("Two-parameter", None),
)
PARAMETER_BUDGET_MODEL_ORDER = (
    "HEA",
    "ZZ",
    "ZZ-CRZ",
    "ZZ evolution",
    "TCNN",
    "Two-parameter",
)


@dataclass(frozen=True)
class WidthRepresentative:
    """The middle fresh-W1 run selected within a single TCNN width."""

    width: int
    run: int
    fresh_marginal_w1: float


def configuration_name(width: int) -> str:
    if width not in TCNN_WIDTHS:
        raise ValueError(f"Unsupported TCNN width: {width}")
    return f"TCNN-{width}"


def cache_key(width: int) -> str:
    if width not in TCNN_WIDTHS:
        raise ValueError(f"Unsupported TCNN width: {width}")
    return f"tcnn_{width}"


def middle_representatives_by_width(
    scores_by_width: Mapping[int, Iterable[tuple[int, float]]],
) -> dict[int, WidthRepresentative]:
    """Rank each width independently and return its deterministic middle run.

    Scores are mean fresh-sample marginal W1 values; lower is better.  Sorting
    also uses run number to make ties reproducible.  The normal three-run sweep
    selects index one; for an even count this deliberately uses the upper
    middle, matching the existing ``len(values) // 2`` convention.
    """

    selected: dict[int, WidthRepresentative] = {}
    for width in TCNN_WIDTHS:
        ranked = sorted(
            ((int(run), float(score)) for run, score in scores_by_width.get(width, ())),
            key=lambda item: (item[1], item[0]),
        )
        if not ranked:
            raise ValueError(f"No fresh-W1 scores supplied for TCNN width {width}")
        run, score = ranked[len(ranked) // 2]
        selected[width] = WidthRepresentative(width, run, score)
    return selected


def select_tcnn_family_representative(
    scores_by_width: Mapping[int, Iterable[tuple[int, float]]],
) -> WidthRepresentative:
    """Choose the TCNN width whose own middle run has the lowest fresh W1."""

    middle = middle_representatives_by_width(scores_by_width)
    return min(
        middle.values(),
        key=lambda item: (item.fresh_marginal_w1, TCNN_WIDTHS.index(item.width)),
    )


def representative_cache_needs_refresh(
    available_keys: Iterable[str],
    cached_provenance: Mapping[str, object],
    expected_provenance: Mapping[str, object],
) -> bool:
    """Return whether the representative-window cache is incomplete or stale."""

    required = set(REQUIRED_REPRESENTATIVE_CACHE_KEYS)
    if not required.issubset(set(available_keys)):
        return True
    return any(cached_provenance.get(key) != expected_provenance.get(key) for key in required)


def parameter_budget_marker(model: str, layer: int | None) -> str:
    """Return the marker with TCNN width routing ahead of quantum layers."""

    if model == "TCNN":
        if layer not in TCNN_WIDTH_MARKERS:
            raise ValueError(f"Unsupported TCNN width for marker routing: {layer}")
        return TCNN_WIDTH_MARKERS[layer]
    if layer is not None:
        return QUANTUM_LAYER_MARKERS[layer]
    if model == "Two-parameter":
        return "X"
    raise ValueError(f"No parameter-budget marker for model={model!r}, layer={layer!r}")
