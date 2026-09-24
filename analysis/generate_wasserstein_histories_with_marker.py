"""Generate Wasserstein-history figures with marked run trends."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter1d

import analyze_four_model_performance as source


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FOLDER = "performance_analysis"
OUTPUT_NAME = "07_wasserstein_loss_vs_epoch_with_marker.png"
SMOOTH_SIGMA = 8.0

RUN_STYLES = {0: "-", 1: "--", 2: ":"}
RUN_MARKERS = {0: "o", 1: "s", 2: "D"}
MARKER_TARGET_EPOCHS = {
    0: (600.0, 1600.0, 2600.0, 3600.0),
    1: (750.0, 1750.0, 2750.0, 3750.0),
    2: (900.0, 1900.0, 2900.0, 3900.0),
}
LAYER_COLORS = {
    1: "#0072B2",
    2: "#E69F00",
    3: "#009E73",
    4: "#7B4FA3",
}


def centered_log_gaussian(
    values: np.ndarray, sigma: float = SMOOTH_SIGMA
) -> np.ndarray:
    """Smooth a positive series symmetrically in log space."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("Expected a non-empty one-dimensional series.")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Gaussian sigma must be finite and positive.")
    if not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise ValueError("Log-Gaussian smoothing requires finite positive values.")
    return np.exp(
        gaussian_filter1d(np.log(array), sigma=float(sigma), mode="nearest")
    )


def marker_indices(epochs: np.ndarray, run: int) -> np.ndarray:
    """Return sparse, run-staggered marker indices nearest target epochs."""
    if run not in MARKER_TARGET_EPOCHS:
        raise ValueError(f"No marker schedule for run {run}.")
    epoch_array = np.asarray(epochs, dtype=float)
    if epoch_array.ndim != 1 or epoch_array.size == 0:
        raise ValueError("Expected a non-empty one-dimensional epoch grid.")
    if not np.all(np.isfinite(epoch_array)) or np.any(np.diff(epoch_array) <= 0):
        raise ValueError("Epoch grid must be finite and strictly increasing.")

    indices = {
        int(np.argmin(np.abs(epoch_array - target)))
        for target in MARKER_TARGET_EPOCHS[run]
        if epoch_array[0] <= target <= epoch_array[-1]
    }
    return np.asarray(sorted(indices), dtype=int)


def variant_output_path(original: Path) -> Path:
    """Append ``_with_marker`` to a source figure path."""
    original = Path(original)
    output = original.with_name(f"{original.stem}_with_marker{original.suffix}")
    if output == original:
        raise RuntimeError("Variant output path would overwrite its source.")
    return output


def valid_trace(item: source.RunMetrics) -> bool:
    return bool(
        item.epochs is not None
        and item.losses is not None
        and len(item.epochs)
        and len(item.losses)
    )


def smooth_run_histories(
    traces: dict[int, source.RunMetrics],
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Interpolate the three runs to one grid and smooth each independently."""
    if set(traces) != set(source.RUNS):
        raise RuntimeError(
            f"Expected runs {list(source.RUNS)}; found {sorted(traces)}."
        )
    if not all(valid_trace(item) for item in traces.values()):
        raise RuntimeError("All three runs must contain a valid training history.")

    common_start = max(float(item.epochs[0]) for item in traces.values())
    common_end = min(float(item.epochs[-1]) for item in traces.values())
    if common_start >= common_end:
        raise RuntimeError("The three run histories have no shared epoch interval.")
    common_epochs = np.unique(
        np.concatenate(
            [
                item.epochs[
                    (item.epochs >= common_start) & (item.epochs <= common_end)
                ]
                for item in traces.values()
            ]
        )
    )
    if common_epochs.size < 2:
        raise RuntimeError("The shared epoch grid is too short to smooth.")

    smoothed = {
        run: centered_log_gaussian(
            np.interp(common_epochs, item.epochs, item.losses)
        )
        for run, item in traces.items()
    }
    return common_epochs, smoothed


def history_panel(
    axis: plt.Axes,
    traces: dict[int, source.RunMetrics],
    color: str,
    layer: int,
    panel_label: str,
) -> None:
    """Draw raw histories, smoothed-run envelope, and three marked trends."""
    for run in source.RUNS:
        item = traces[run]
        axis.plot(
            item.epochs,
            item.losses,
            color=color,
            linestyle=RUN_STYLES[run],
            linewidth=0.7,
            alpha=0.22,
            zorder=1,
        )

    epochs, smoothed = smooth_run_histories(traces)
    stacked = np.vstack([smoothed[run] for run in source.RUNS])
    axis.fill_between(
        epochs,
        np.min(stacked, axis=0),
        np.max(stacked, axis=0),
        color=color,
        alpha=0.16,
        linewidth=0,
        zorder=2,
    )

    for run in source.RUNS:
        trend = smoothed[run]
        axis.plot(
            epochs,
            trend,
            color=color,
            linestyle="-",
            linewidth=1.8,
            zorder=3,
        )
        indices = marker_indices(epochs, run)
        axis.plot(
            epochs[indices],
            trend[indices],
            linestyle="none",
            marker=RUN_MARKERS[run],
            markersize=5.4,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.15,
            zorder=4,
        )

    axis.set_title(f"Layer {layer}", pad=4)
    axis.set_yscale("log")
    axis.set_xlim(0, 4250)
    axis.grid(True, which="major", alpha=0.24)
    axis.grid(True, which="minor", axis="y", alpha=0.08)
    axis.tick_params(direction="out", length=3.0, width=0.7)
    axis.text(
        -0.08,
        1.04,
        f"({panel_label})",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )


def history_legend(figure: plt.Figure) -> None:
    heading = lambda label: Line2D(  # noqa: E731 - compact legend-only handle
        [], [], color="none", linestyle="none", label=label
    )
    raw_handles = [
        Line2D(
            [0],
            [0],
            color="#555555",
            linestyle=RUN_STYLES[run],
            linewidth=0.9,
            alpha=0.55,
            label=f"Run {run}",
        )
        for run in source.RUNS
    ]
    range_handle = Patch(
        facecolor="#777777",
        alpha=0.16,
        label="Min–max envelope",
    )
    smooth_handles = [
        Line2D(
            [0],
            [0],
            color="#444444",
            linestyle="-",
            linewidth=1.8,
            marker=RUN_MARKERS[run],
            markersize=5.4,
            markerfacecolor="white",
            markeredgecolor="#444444",
            markeredgewidth=1.15,
            label=f"Run {run}",
        )
        for run in source.RUNS
    ]
    raw_legend = figure.legend(
        handles=[heading("Raw histories:"), *raw_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.044),
        ncol=4,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.55,
        fontsize=10.2,
    )
    smooth_legend = figure.legend(
        handles=[heading("Smoothed trends:"), *smooth_handles, range_handle],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.006),
        ncol=5,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.45,
        fontsize=10.2,
    )
    raw_legend.get_texts()[0].set_fontweight("semibold")
    smooth_legend.get_texts()[0].set_fontweight("semibold")


def plot_model(
    model_name: str,
    metrics: dict[tuple[int, int], source.RunMetrics],
    output: Path,
) -> None:
    figure, axes = plt.subplots(
        2, 2, figsize=(13, 9.4), sharex=True, sharey=True
    )
    for layer, axis, panel_label in zip(source.LAYERS, axes.flat, "abcd"):
        traces = {run: metrics[(layer, run)] for run in source.RUNS}
        history_panel(
            axis,
            traces,
            LAYER_COLORS[layer],
            layer,
            panel_label,
        )

    figure.supxlabel("Epoch", y=0.115)
    figure.supylabel("Marginal Wasserstein distance (log scale)", x=0.02)
    figure.suptitle(
        f"{model_name}: Wasserstein training histories\n"
        rf"centered log-Gaussian trends ($\sigma={SMOOTH_SIGMA:g}$ logged points)",
        fontsize=15,
        y=0.985,
    )
    history_legend(figure)
    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        top=0.90,
        bottom=0.21,
        wspace=0.13,
        hspace=0.22,
    )
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def generate_all() -> list[dict[str, object]]:
    from analyze_four_quantum_model_complete import load_run_metrics_complete

    records: list[dict[str, object]] = []
    for model_name, sweep in source.SWEEPS.items():
        metrics, notes = load_run_metrics_complete(sweep)
        unexpected = [
            note for note in notes if "explicit user confirmation" not in note
        ]
        if unexpected:
            raise RuntimeError(
                f"Unexpected complete-history notes for {model_name}: {unexpected}"
            )
        missing = [
            (layer, run)
            for layer in source.LAYERS
            for run in source.RUNS
            if (layer, run) not in metrics or not valid_trace(metrics[(layer, run)])
        ]
        if missing:
            raise RuntimeError(f"Missing histories for {model_name}: {missing}")

        output = sweep.parent / OUTPUT_FOLDER / OUTPUT_NAME
        output.parent.mkdir(parents=True, exist_ok=True)
        plot_model(model_name, metrics, output)
        records.append(
            {
                "model": model_name,
                "output": str(output.relative_to(ROOT)),
                "smoother": "centered log-Gaussian",
                "sigma_logged_points": SMOOTH_SIGMA,
                "runs": list(source.RUNS),
            }
        )
    return records


def main() -> None:
    source.configure_plotting()
    print(json.dumps(generate_all(), indent=2))


if __name__ == "__main__":
    main()
