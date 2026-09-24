"""Post-training analysis for the TCNN and two-parameter WGAN baselines.

Run this script after both classical sweeps have completed.  By default it
discovers the newest matching sweep folders in classical_baseline, requires all
three runs to reach epoch 4000, and writes to its performance_analysis folder.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde, wasserstein_distance
from tqdm.auto import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent / "classical_baseline"
OUTPUT_DIR = BASE_DIR / "performance_analysis"


def project_path(path: Path) -> str:
    """Write paths relative to the delivered project when possible."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(BASE_DIR.parent).as_posix()
    except ValueError:
        return str(resolved)


if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import data_handling as dh
from tcnn_family import (
    TCNN_CONFIGURATIONS,
    TCNN_FAMILY_COLOR,
    TCNN_WIDTHS,
    select_tcnn_family_representative,
)


WIDTHS = TCNN_WIDTHS
RUNS = (0, 1, 2)
LAGS = np.arange(1, 5)
EMA_SPAN = 41
CONFIG_ORDER = (*TCNN_CONFIGURATIONS, "Two-parameter")
COLORS = {
    **{configuration: TCNN_FAMILY_COLOR for configuration in TCNN_CONFIGURATIONS},
    "Two-parameter": "#e45756",
}
QUANTILES = np.linspace(0.0025, 0.9975, 240)
METRIC_NAMES = ("abs_acf", "acf", "leverage")
METRIC_TITLES = {
    "abs_acf": "Absolute-return ACF",
    "acf": "Raw-return ACF",
    "leverage": "Leverage",
}


@dataclass
class RunRecord:
    configuration: str
    family: str
    width: int | None
    run: int
    seed: int
    generator_parameters: int
    best_wasserstein: float
    best_epoch: int
    checkpoint_batch_wasserstein: float
    fresh_wasserstein_mean: float
    fresh_wasserstein_median: float
    fresh_wasserstein_std: float
    fresh_wasserstein_min: float
    fresh_wasserstein_max: float
    qq_rmse: float
    abs_acf_mae: float
    acf_mae: float
    leverage_mae: float
    loss_epochs: np.ndarray
    loss_wasserstein: np.ndarray
    generated_windows: np.ndarray
    metric_distributions: dict[str, np.ndarray]
    metric_centers: dict[str, np.ndarray]

    def table_row(self) -> dict[str, object]:
        return {
            "configuration": self.configuration,
            "family": self.family,
            "width": self.width,
            "run": self.run,
            "seed": self.seed,
            "generator_parameters": self.generator_parameters,
            "best_wasserstein": self.best_wasserstein,
            "best_epoch": self.best_epoch,
            "checkpoint_selecting_batch_wasserstein": self.checkpoint_batch_wasserstein,
            "evaluation_marginal_wasserstein": self.fresh_wasserstein_mean,
            "fresh_wasserstein_median": self.fresh_wasserstein_median,
            "fresh_wasserstein_std": self.fresh_wasserstein_std,
            "fresh_wasserstein_min": self.fresh_wasserstein_min,
            "fresh_wasserstein_max": self.fresh_wasserstein_max,
            "qq_rmse": self.qq_rmse,
            "abs_acf_discrepancy_mae_lags_1_4": self.abs_acf_mae,
            "acf_discrepancy_mae_lags_1_4": self.acf_mae,
            "leverage_discrepancy_mae_lags_1_4": self.leverage_mae,
            "n_generated_windows": len(self.generated_windows),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tcnn-sweep", type=Path)
    parser.add_argument("--two-parameter-sweep", type=Path)
    parser.add_argument("--expected-final-epoch", type=int, default=4000)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-block-length", type=int, default=64)
    parser.add_argument("--fresh-evaluation-samples", type=int, default=10000)
    parser.add_argument("--fresh-evaluation-repeats", type=int, default=20)
    parser.add_argument("--fresh-evaluation-seed", type=int, default=20260831)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Analyze available completed configurations instead of requiring every configured run.",
    )
    parser.add_argument(
        "--readiness-check",
        action="store_true",
        help="Report expected run artifacts and completion without evaluating checkpoints.",
    )
    return parser.parse_args()


def newest_sweep(pattern: str) -> Path:
    candidates = [path for path in BASE_DIR.glob(pattern) if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(
            f"No folder matching {pattern!r} was found in {BASE_DIR}. "
            "Pass the sweep explicitly on the command line."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_sweep(explicit: Path | None, pattern: str) -> Path:
    if explicit is None:
        return newest_sweep(pattern).resolve()
    return (BASE_DIR / explicit).resolve() if not explicit.is_absolute() else explicit.resolve()


def load_text_array(path: Path, dtype=float) -> np.ndarray:
    values = np.loadtxt(path, dtype=dtype)
    return np.atleast_1d(values)


def window_metric_distributions(windows: np.ndarray) -> dict[str, np.ndarray]:
    """Use the same window-respecting estimator as the quantum-model analysis."""

    windows = np.asarray(windows, dtype=float)
    if windows.ndim != 2 or windows.shape[1] != 16:
        raise ValueError(f"Expected generated windows with shape (n, 16), got {windows.shape}")
    result = {
        name: np.full((len(windows), len(LAGS)), np.nan, dtype=float)
        for name in METRIC_NAMES
    }
    # The global centers are deliberately shared across windows. This is the
    # estimator used for all four quantum families in the comparison tables.
    raw = windows - np.mean(windows)
    absolute = np.abs(windows) - np.mean(np.abs(windows))
    squared = windows**2 - np.mean(windows**2)
    raw_variance = np.sum(raw**2, axis=1)
    absolute_variance = np.sum(absolute**2, axis=1)
    squared_variance = np.sum(squared**2, axis=1)

    for lag_index, lag in enumerate(LAGS):
        correction = windows.shape[1] / (windows.shape[1] - lag)
        raw_numerator = np.sum(raw[:, :-lag] * raw[:, lag:], axis=1)
        absolute_numerator = np.sum(
            absolute[:, :-lag] * absolute[:, lag:], axis=1
        )
        leverage_numerator = np.sum(raw[:, :-lag] * squared[:, lag:], axis=1)
        np.divide(
            correction * raw_numerator,
            raw_variance,
            out=result["acf"][:, lag_index],
            where=raw_variance > 0,
        )
        np.divide(
            correction * absolute_numerator,
            absolute_variance,
            out=result["abs_acf"][:, lag_index],
            where=absolute_variance > 0,
        )
        leverage_denominator = np.sqrt(raw_variance * squared_variance)
        np.divide(
            correction * leverage_numerator,
            leverage_denominator,
            out=result["leverage"][:, lag_index],
            where=leverage_denominator > 0,
        )
    return result


def metric_centers(distributions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        name: np.nanmedian(values, axis=0)
        for name, values in distributions.items()
    }


def qq_rmse(generated: np.ndarray, real: np.ndarray) -> float:
    generated_quantiles = np.quantile(generated, QUANTILES)
    real_quantiles = np.quantile(real, QUANTILES)
    return float(np.sqrt(np.mean((generated_quantiles - real_quantiles) ** 2)))


def completion_rows(
    tcnn_sweep: Path,
    two_parameter_sweep: Path,
    expected_final_epoch: int,
) -> list[dict[str, object]]:
    expected = []
    for width in WIDTHS:
        config_dir = tcnn_sweep / f"filters_{width}_window_16_stride_1"
        for run in RUNS:
            expected.append((f"TCNN-{width}", "TCNN", width, run, config_dir / f"run_{run}"))
    config_dir = two_parameter_sweep / "two_parameter_window_16_stride_1"
    for run in RUNS:
        expected.append(("Two-parameter", "two-parameter", None, run, config_dir / f"run_{run}"))

    rows = []
    for configuration, family, width, run, run_dir in expected:
        epochs_path = run_dir / "metrics" / "loss_epochs.txt"
        best_path = run_dir / "metrics" / "best_metrics.txt"
        generated_path = run_dir / "generated_windows_best.npy"
        generator_weights_path = run_dir / "weights" / "generator_lowest_wass.pkl"
        manifest_path = run_dir / "experiment_manifest.json"
        last_epoch = np.nan
        if epochs_path.exists():
            try:
                epochs = load_text_array(epochs_path)
                if len(epochs):
                    last_epoch = int(epochs[-1])
            except (OSError, ValueError):
                pass
        complete = bool(
            run_dir.is_dir()
            and best_path.exists()
            and generator_weights_path.exists()
            and generated_path.exists()
            and manifest_path.exists()
            and np.isfinite(last_epoch)
            and last_epoch >= expected_final_epoch
        )
        rows.append(
            {
                "configuration": configuration,
                "family": family,
                "width": width,
                "run": run,
                "run_directory": str(run_dir),
                "last_evaluation_epoch": last_epoch,
                "required_final_epoch": expected_final_epoch,
                "best_metrics_available": best_path.exists(),
                "best_generator_weights_available": generator_weights_path.exists(),
                "generated_windows_available": generated_path.exists(),
                "manifest_available": manifest_path.exists(),
                "complete": complete,
            }
        )
    return rows


def fresh_generator_evaluation(
    row: dict[str, object],
    real_returns: np.ndarray,
    transform_params: list[float],
    scale_factor: float,
    samples: int,
    repeats: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a saved generator independently of checkpoint selection."""

    from classical_wgan_common import (  # Lazy import keeps --help lightweight.
        make_tcnn_generator,
        make_two_parameter_generator,
    )

    run_dir = Path(str(row["run_directory"]))
    if row["family"] == "TCNN":
        generator = make_tcnn_generator(16, int(row["width"]))
    else:
        generator = make_two_parameter_generator(16)
    with (run_dir / "weights" / "generator_lowest_wass.pkl").open("rb") as handle:
        generator.set_weights(pickle.load(handle))

    rng = np.random.default_rng(seed)
    distances = np.empty(repeats, dtype=float)
    diagnostic_windows = None
    for repeat in range(repeats):
        noise = rng.uniform(-np.pi, np.pi, size=(samples, 16))
        generated_scaled = generator(noise, training=False).numpy()
        generated = dh.inverse_transform(
            generated_scaled * scale_factor, transform_params
        )
        distances[repeat] = wasserstein_distance(
            generated.reshape(-1), real_returns.reshape(-1)
        )
        if diagnostic_windows is None:
            diagnostic_windows = generated
    if diagnostic_windows is None:
        raise ValueError("fresh-evaluation-repeats must be positive")
    return diagnostic_windows, distances


def load_completed_record(
    row: dict[str, object],
    real_returns: np.ndarray,
    real_centers: dict[str, np.ndarray],
    transform_params: list[float],
    scale_factor: float,
    fresh_samples: int,
    fresh_repeats: int,
    fresh_seed: int,
) -> RunRecord:
    run_dir = Path(str(row["run_directory"]))
    with (run_dir / "experiment_manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    best = load_text_array(run_dir / "metrics" / "best_metrics.txt")
    loss_epochs = load_text_array(run_dir / "metrics" / "loss_epochs.txt")
    loss_wasserstein = load_text_array(run_dir / "metrics" / "loss_wass.txt")
    common_length = min(len(loss_epochs), len(loss_wasserstein))
    loss_epochs = loss_epochs[:common_length]
    loss_wasserstein = loss_wasserstein[:common_length]
    checkpoint_generated = np.load(run_dir / "generated_windows_best.npy")
    generated, fresh_distances = fresh_generator_evaluation(
        row,
        real_returns,
        transform_params,
        scale_factor,
        fresh_samples,
        fresh_repeats,
        fresh_seed,
    )
    distributions = window_metric_distributions(generated)
    centers = metric_centers(distributions)
    discrepancies = {
        name: float(np.nanmean(np.abs(centers[name] - real_centers[name])))
        for name in METRIC_NAMES
    }
    return RunRecord(
        configuration=str(row["configuration"]),
        family=str(row["family"]),
        width=int(row["width"]) if pd.notna(row["width"]) else None,
        run=int(row["run"]),
        seed=int(manifest["seed"]),
        generator_parameters=int(manifest["generator_parameter_count"]),
        best_wasserstein=float(best[0]),
        best_epoch=int(best[1]),
        checkpoint_batch_wasserstein=float(
            wasserstein_distance(
                checkpoint_generated.reshape(-1), real_returns.reshape(-1)
            )
        ),
        fresh_wasserstein_mean=float(np.mean(fresh_distances)),
        fresh_wasserstein_median=float(np.median(fresh_distances)),
        fresh_wasserstein_std=float(np.std(fresh_distances, ddof=1)),
        fresh_wasserstein_min=float(np.min(fresh_distances)),
        fresh_wasserstein_max=float(np.max(fresh_distances)),
        qq_rmse=qq_rmse(generated.reshape(-1), real_returns.reshape(-1)),
        abs_acf_mae=discrepancies["abs_acf"],
        acf_mae=discrepancies["acf"],
        leverage_mae=discrepancies["leverage"],
        loss_epochs=loss_epochs,
        loss_wasserstein=loss_wasserstein,
        generated_windows=generated,
        metric_distributions=distributions,
        metric_centers=centers,
    )


def role_map(records: Iterable[RunRecord]) -> dict[str, dict[str, RunRecord]]:
    roles = {}
    for configuration in CONFIG_ORDER:
        available = sorted(
            [record for record in records if record.configuration == configuration],
            key=lambda record: record.fresh_wasserstein_mean,
        )
        if len(available) == 3:
            roles[configuration] = {
                "best": available[0],
                "middle": available[1],
                "worst": available[2],
            }
    return roles


def tcnn_family_representative(records: Iterable[RunRecord]) -> RunRecord:
    """Return the width-level middle run with the lowest fresh marginal W1."""

    materialized = list(records)
    scores_by_width = {
        width: [
            (record.run, record.fresh_wasserstein_mean)
            for record in materialized
            if record.family == "TCNN" and record.width == width
        ]
        for width in WIDTHS
    }
    selected = select_tcnn_family_representative(scores_by_width)
    for record in materialized:
        if record.width == selected.width and record.run == selected.run:
            return record
    raise RuntimeError("Selected TCNN family representative was not present in records")


def simultaneous_interval(center: np.ndarray, estimates: np.ndarray):
    standard_error = np.nanstd(estimates, axis=0, ddof=1)
    usable = np.isfinite(standard_error) & (standard_error > np.finfo(float).eps)
    lower = np.array(center, copy=True)
    upper = np.array(center, copy=True)
    if np.any(usable):
        standardized = np.abs(
            (estimates[:, usable] - center[usable]) / standard_error[usable]
        )
        critical = float(np.nanquantile(np.nanmax(standardized, axis=1), 0.95))
        lower[usable] = center[usable] - critical * standard_error[usable]
        upper[usable] = center[usable] + critical * standard_error[usable]
    return lower, upper


def moving_block_sample(values, block_length, rng):
    block_length = min(block_length, len(values))
    count = int(np.ceil(len(values) / block_length))
    starts = rng.integers(0, len(values) - block_length + 1, size=count)
    return np.concatenate(
        [values[start : start + block_length] for start in starts]
    )[: len(values)]


def bootstrap_real_intervals(real, centers, replicates, block_length):
    estimates = {
        name: np.empty((replicates, len(LAGS)), dtype=float)
        for name in METRIC_NAMES
    }
    rng = np.random.default_rng(20260828)
    progress = tqdm(range(replicates), desc="real-data bootstrap", unit="draw")
    for draw in progress:
        sampled = moving_block_sample(real, block_length, rng)
        windows = dh.chopchop(sampled, 16, 1)
        sampled_centers = metric_centers(window_metric_distributions(windows))
        for name in METRIC_NAMES:
            estimates[name][draw] = sampled_centers[name]
    return {
        name: simultaneous_interval(centers[name], estimates[name])
        for name in METRIC_NAMES
    }


def bootstrap_generated_intervals(record, replicates, seed):
    rng = np.random.default_rng(seed)
    result = {}
    for name, values in record.metric_distributions.items():
        estimates = np.empty((replicates, len(LAGS)), dtype=float)
        for start in range(0, replicates, 25):
            stop = min(start + 25, replicates)
            indices = rng.integers(0, len(values), size=(stop - start, len(values)))
            estimates[start:stop] = np.nanmedian(values[indices], axis=1)
        result[name] = simultaneous_interval(record.metric_centers[name], estimates)
    return result


def save_figure(figure: plt.Figure, png_path: Path) -> None:
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def configured_records(records, roles):
    return [roles[name]["middle"] for name in CONFIG_ORDER if name in roles]


def plot_distribution(real, records, output):
    flattened = [real] + [record.generated_windows.reshape(-1) for record in records]
    combined = np.concatenate(flattened)
    low, high = np.quantile(combined, [0.001, 0.999])
    grid = np.linspace(low, high, 700)
    figure, axis = plt.subplots(figsize=(10.5, 6.4))
    axis.hist(
        real,
        bins=70,
        range=(low, high),
        density=True,
        color="#9d9da1",
        alpha=0.55,
        label="Bitcoin",
    )
    for record in records:
        values = record.generated_windows.reshape(-1)
        try:
            density = gaussian_kde(values)(grid)
            axis.plot(
                grid,
                density,
                color=COLORS[record.configuration],
                linewidth=2.2,
                label=record.configuration,
            )
        except np.linalg.LinAlgError:
            continue
    axis.set(title="Classical WGAN return distributions", xlabel="Log return", ylabel="Density")
    axis.legend(ncol=2)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    save_figure(figure, output)


def plot_qq(real, records, output):
    real_quantiles = np.quantile(real, QUANTILES)
    combined = [real_quantiles]
    figure, axis = plt.subplots(figsize=(8.2, 7.2))
    for record in records:
        generated_quantiles = np.quantile(record.generated_windows, QUANTILES)
        combined.append(generated_quantiles)
        axis.plot(
            real_quantiles,
            generated_quantiles,
            color=COLORS[record.configuration],
            linewidth=2,
            label=record.configuration,
        )
    limits = [min(np.min(item) for item in combined), max(np.max(item) for item in combined)]
    axis.plot(limits, limits, "--", color="black", linewidth=1.4, label="45° reference")
    axis.set(
        title="Q-Q plot: generated versus Bitcoin",
        xlabel="Bitcoin quantile",
        ylabel="Generated quantile",
        xlim=limits,
        ylim=limits,
    )
    axis.set_aspect("equal", adjustable="box")
    axis.legend(fontsize=9)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    save_figure(figure, output)


def plot_dependence(metric, real_centers, real_intervals, records, generated_intervals, output):
    figure, axis = plt.subplots(figsize=(10.5, 6.4))
    axis.axhline(0.0, color="black", linewidth=1.8, zorder=1)
    real_lower, real_upper = real_intervals[metric]
    axis.fill_between(LAGS, real_lower, real_upper, color="#555555", alpha=0.13)
    axis.plot(
        LAGS,
        real_centers[metric],
        "--D",
        color="#222222",
        linewidth=2.5,
        label="Bitcoin",
        zorder=4,
    )
    for record in records:
        color = COLORS[record.configuration]
        lower, upper = generated_intervals[record.configuration][metric]
        axis.fill_between(LAGS, lower, upper, color=color, alpha=0.08)
        axis.plot(
            LAGS,
            record.metric_centers[metric],
            marker="o",
            color=color,
            linewidth=2.2,
            label=record.configuration,
            zorder=3,
        )
    axis.set(
        title=f"Classical WGAN: {METRIC_TITLES[metric]}",
        xlabel="Lag",
        ylabel="Correlation",
        xticks=LAGS,
    )
    axis.legend(ncol=2, fontsize=9)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    save_figure(figure, output)


def plot_absolute_acf_violin(real_distributions, records, output):
    rows = [("Bitcoin", real_distributions["abs_acf"], "#555555")]
    rows.extend(
        (
            record.configuration,
            record.metric_distributions["abs_acf"],
            COLORS[record.configuration],
        )
        for record in records
    )
    figure, axes = plt.subplots(len(rows), 1, figsize=(10.5, 2.25 * len(rows)), sharex=True)
    for axis, (label, values, color) in zip(axes, rows):
        usable = [values[np.isfinite(values[:, index]), index] for index in range(len(LAGS))]
        violin = axis.violinplot(usable, positions=LAGS, widths=0.72, showmedians=True)
        for body in violin["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.38)
        for key in ("cmins", "cmaxes", "cbars", "cmedians"):
            violin[key].set_color(color)
            violin[key].set_linewidth(1.2)
        axis.axhline(0, color="black", linewidth=1)
        axis.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=42)
        axis.grid(alpha=0.15)
    axes[-1].set(xlabel="Lag", xticks=LAGS)
    figure.suptitle("Window-level absolute-return ACF distributions", y=0.995)
    figure.tight_layout()
    save_figure(figure, output)


def ema(values, span=EMA_SPAN):
    return pd.Series(values).ewm(span=span, adjust=False).mean().to_numpy()


def plot_wasserstein_histories(roles, output):
    configurations = [name for name in CONFIG_ORDER if name in roles]
    ncols = min(3, len(configurations))
    nrows = int(np.ceil(len(configurations) / ncols))
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.4 * ncols, 5.0 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes.reshape(-1)
    for axis, configuration in zip(axes, configurations):
        group = roles[configuration]
        color = COLORS[configuration]
        for role in ("best", "middle", "worst"):
            record = group[role]
            axis.plot(
                record.loss_epochs,
                record.loss_wasserstein,
                color=color,
                linewidth=0.75,
                alpha=0.18,
            )
        common_epochs = group["middle"].loss_epochs
        smoothed = {}
        for role in ("best", "middle", "worst"):
            record = group[role]
            interpolated = np.interp(common_epochs, record.loss_epochs, record.loss_wasserstein)
            smoothed[role] = ema(interpolated)
        axis.fill_between(
            common_epochs,
            np.minimum(smoothed["best"], smoothed["worst"]),
            np.maximum(smoothed["best"], smoothed["worst"]),
            color=color,
            alpha=0.18,
        )
        axis.plot(common_epochs, smoothed["middle"], color=color, linewidth=3)
        axis.set_title(configuration)
        axis.set_yscale("log")
        axis.grid(alpha=0.2, which="both")
    for axis in axes[len(configurations) :]:
        axis.axis("off")
    figure.supxlabel("Epoch")
    figure.supylabel("Marginal Wasserstein distance (log scale)")
    figure.suptitle("Classical WGAN training histories")
    figure.legend(
        handles=[
            Line2D([0], [0], color="#4c78a8", alpha=0.2, label="raw run histories"),
            Patch(facecolor="#4c78a8", alpha=0.18, label="best–worst smoothed envelope"),
            Line2D([0], [0], color="#4c78a8", linewidth=3, label=f"middle run, EMA span {EMA_SPAN}"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=3,
        frameon=False,
    )
    figure.tight_layout(rect=(0.03, 0.03, 1, 0.87))
    save_figure(figure, output)


def plot_two_budget_comparison(records, output):
    definitions = (
        ("fresh_wasserstein_mean", "Fresh-sample marginal Wasserstein distance"),
        ("abs_acf_mae", "Absolute-ACF MAE, lags 1–4"),
        ("acf_mae", "Raw-ACF MAE, lags 1–4"),
        ("leverage_mae", "Leverage MAE, lags 1–4"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.2), sharex=True)
    for axis, (attribute, title) in zip(axes.reshape(-1), definitions):
        for configuration in CONFIG_ORDER:
            group = [record for record in records if record.configuration == configuration]
            if not group:
                continue
            x = group[0].generator_parameters
            values = np.asarray([getattr(record, attribute) for record in group])
            axis.scatter(
                np.full(len(values), x),
                values,
                color=COLORS[configuration],
                alpha=0.35,
                s=36,
            )
            median = float(np.median(values))
            marker = "D" if configuration == "Two-parameter" else "o"
            axis.scatter(x, median, color=COLORS[configuration], marker=marker, s=90, edgecolor="black", linewidth=0.6)
        axis.set_xscale("log")
        axis.set_title(title)
        axis.grid(alpha=0.2, which="both")
    figure.supxlabel("Trainable generator parameters (log scale)")
    figure.supylabel("Discrepancy (lower is better)")
    figure.suptitle("Compact classical generators at two parameter budgets", y=0.99)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS[name], markeredgecolor="black", label=name)
        for name in CONFIG_ORDER
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=len(handles),
        frameon=False,
    )
    figure.tight_layout(rect=(0.03, 0.04, 1, 0.90))
    save_figure(figure, output)


def plot_seed_sensitivity(records, output):
    definitions = (
        ("fresh_wasserstein_mean", "Fresh-sample marginal W1"),
        ("abs_acf_mae", "Absolute-ACF MAE"),
        ("acf_mae", "Raw-ACF MAE"),
        ("leverage_mae", "Leverage MAE"),
    )
    figure, axes = plt.subplots(1, 4, figsize=(17, 5.2))
    positions = np.arange(1, len(CONFIG_ORDER) + 1)
    for axis, (attribute, title) in zip(axes, definitions):
        data = [
            [getattr(record, attribute) for record in records if record.configuration == configuration]
            for configuration in CONFIG_ORDER
        ]
        axis.boxplot(data, positions=positions, widths=0.55, showfliers=False)
        for position, configuration, values in zip(positions, CONFIG_ORDER, data):
            axis.scatter(
                np.full(len(values), position),
                values,
                color=COLORS[configuration],
                s=36,
                zorder=3,
            )
        axis.set_title(title)
        axis.set_xticks(positions, CONFIG_ORDER, rotation=35, ha="right")
        axis.grid(alpha=0.2, axis="y")
    figure.suptitle("Seed sensitivity across completed classical runs")
    figure.tight_layout()
    save_figure(figure, output)


def write_tables(records, roles, output):
    all_rows = []
    for record in records:
        row = record.table_row()
        row["performance_role"] = next(
            (
                role
                for role, selected in roles.get(record.configuration, {}).items()
                if selected.run == record.run
            ),
            "unranked",
        )
        all_rows.append(row)
    all_frame = pd.DataFrame(all_rows)
    all_frame.to_csv(output / "all_run_metrics.csv", index=False)

    metric_columns = [
        "best_wasserstein",
        "checkpoint_selecting_batch_wasserstein",
        "evaluation_marginal_wasserstein",
        "fresh_wasserstein_std",
        "qq_rmse",
        "abs_acf_discrepancy_mae_lags_1_4",
        "acf_discrepancy_mae_lags_1_4",
        "leverage_discrepancy_mae_lags_1_4",
    ]
    summary_rows = []
    for configuration in CONFIG_ORDER:
        group = all_frame[all_frame["configuration"] == configuration]
        if group.empty:
            continue
        row = {
            "configuration": configuration,
            "generator_parameters": int(group["generator_parameters"].iloc[0]),
            "n_completed_runs": len(group),
            "middle_run": roles[configuration]["middle"].run if configuration in roles else np.nan,
        }
        for column in metric_columns:
            row[f"{column}_median"] = group[column].median()
            row[f"{column}_min"] = group[column].min()
            row[f"{column}_max"] = group[column].max()
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(output / "configuration_summary.csv", index=False)

    middle_rows = [roles[name]["middle"].table_row() for name in CONFIG_ORDER if name in roles]
    pd.DataFrame(middle_rows).to_csv(output / "middle_run_diagnostic_summary.csv", index=False)
    family_record = tcnn_family_representative(records)
    family_row = family_record.table_row()
    family_row["family_configuration"] = "TCNN"
    family_row["selection_rule"] = (
        "lowest fresh marginal W1 among independently ranked width-level middle runs"
    )
    pd.DataFrame([family_row]).to_csv(
        output / "tcnn_family_representative.csv", index=False
    )
    return family_record


def write_analysis_summary(records, roles, output):
    lines = [
        "# Classical baseline analysis",
        "",
        "TCNN is a four-width classical WGAN family (widths 2, 4, 6, and 8). "
        "Each width is ranked independently by fresh-sample marginal W1; a "
        "family-level representative is the width whose middle run has the lowest "
        "fresh-sample marginal W1. The sweep is a bounded parameter-budget study, "
        "not an optimized classical scaling frontier.",
        "",
        "All reported evaluation Wasserstein distances are means over independent "
        "fresh samples drawn from each saved best-Wasserstein generator checkpoint. "
        "The batch used during training to select that checkpoint is retained only "
        "as an audit column.",
        "",
        "The Q-Q RMSE and window-respecting absolute-ACF, raw-ACF, and leverage "
        "estimators use the same quantile grid and formulas as the quantum-model "
        "analysis.",
        "",
        "## Representative runs",
        "",
        "| Configuration | Parameters | Best run | Middle run | Worst run | "
        "Middle fresh W1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for configuration in CONFIG_ORDER:
        if configuration not in roles:
            continue
        group = roles[configuration]
        middle = group["middle"]
        lines.append(
            f"| {configuration} | {middle.generator_parameters} | "
            f"{group['best'].run} | {middle.run} | {group['worst'].run} | "
            f"{middle.fresh_wasserstein_mean:.8g} |"
        )
    (output / "analysis_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    tcnn_sweep = resolve_sweep(args.tcnn_sweep, "sweep_tcnn_btc_daily_2020_2026_*")
    two_sweep = resolve_sweep(
        args.two_parameter_sweep, "sweep_two_parameter_btc_daily_2020_2026_*"
    )
    if args.readiness_check:
        completion = completion_rows(tcnn_sweep, two_sweep, args.expected_final_epoch)
        incomplete = [row for row in completion if not row["complete"]]
        print(
            json.dumps(
                {
                    "tcnn_sweep": str(tcnn_sweep),
                    "two_parameter_sweep": str(two_sweep),
                    "expected_final_epoch": args.expected_final_epoch,
                    "expected_runs": len(completion),
                    "complete_runs": len(completion) - len(incomplete),
                    "ready": not incomplete,
                    "incomplete": incomplete,
                    "runs": completion,
                },
                indent=2,
            )
        )
        return

    output = OUTPUT_DIR
    output.mkdir(parents=True, exist_ok=True)

    completion = completion_rows(tcnn_sweep, two_sweep, args.expected_final_epoch)
    completion_frame = pd.DataFrame(completion)
    delivery_completion = completion_frame.copy()
    delivery_completion["run_directory"] = delivery_completion["run_directory"].map(
        lambda value: project_path(Path(value))
    )
    delivery_completion.to_csv(output / "completion_report.csv", index=False)
    incomplete = completion_frame[~completion_frame["complete"]]
    if not incomplete.empty and not args.allow_incomplete:
        missing = ", ".join(
            f"{row.configuration}/run_{row.run}"
            for row in incomplete.itertuples()
        )
        raise RuntimeError(
            f"Analysis stopped because {len(incomplete)} required runs are incomplete: {missing}. "
            f"See {output / 'completion_report.csv'}, or use --allow-incomplete."
        )

    real_returns = dh.load_BTC_lr()
    transformed, transform_params = dh.transform(real_returns, unity_transform=False)
    scale_factor = float(np.max(np.abs(transformed)))
    real_windows = dh.chopchop(real_returns, 16, 1)
    real_distributions = window_metric_distributions(real_windows)
    real_centers = metric_centers(real_distributions)

    records = []
    completed_rows = [row for row in completion if row["complete"]]
    for index, row in enumerate(
        tqdm(completed_rows, desc="fresh checkpoint evaluation", unit="run")
    ):
        records.append(
            load_completed_record(
                row,
                real_returns,
                real_centers,
                transform_params,
                scale_factor,
                args.fresh_evaluation_samples,
                args.fresh_evaluation_repeats,
                args.fresh_evaluation_seed + index,
            )
        )
    roles = role_map(records)
    if not roles:
        raise RuntimeError("No configuration has three completed runs to analyze.")
    middle_records = configured_records(records, roles)

    family_representative = write_tables(records, roles, output)
    write_analysis_summary(records, roles, output)
    plot_distribution(real_returns, middle_records, output / "01_return_distribution.png")
    plot_qq(real_returns, middle_records, output / "02_qq_plot.png")

    real_intervals = bootstrap_real_intervals(
        real_returns,
        real_centers,
        args.bootstrap_replicates,
        args.bootstrap_block_length,
    )
    generated_intervals = {}
    for index, record in enumerate(
        tqdm(middle_records, desc="generated-window bootstrap", unit="configuration")
    ):
        generated_intervals[record.configuration] = bootstrap_generated_intervals(
            record, args.bootstrap_replicates, seed=20260828 + index
        )
    plot_dependence(
        "abs_acf",
        real_centers,
        real_intervals,
        middle_records,
        generated_intervals,
        output / "03_absolute_acf.png",
    )
    plot_absolute_acf_violin(
        real_distributions,
        middle_records,
        output / "04_absolute_acf_violin.png",
    )
    plot_dependence(
        "acf",
        real_centers,
        real_intervals,
        middle_records,
        generated_intervals,
        output / "05_acf.png",
    )
    plot_dependence(
        "leverage",
        real_centers,
        real_intervals,
        middle_records,
        generated_intervals,
        output / "06_leverage.png",
    )
    plot_wasserstein_histories(roles, output / "07_wasserstein_loss_vs_epoch.png")
    plot_two_budget_comparison(records, output / "08_two_budget_comparison.png")
    plot_seed_sensitivity(records, output / "09_seed_sensitivity.png")

    manifest = {
        "created": datetime.now().isoformat(),
        "tcnn_sweep": project_path(tcnn_sweep),
        "two_parameter_sweep": project_path(two_sweep),
        "expected_final_epoch": args.expected_final_epoch,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_block_length": args.bootstrap_block_length,
        "fresh_evaluation_samples_per_repeat": args.fresh_evaluation_samples,
        "fresh_evaluation_repeats": args.fresh_evaluation_repeats,
        "fresh_evaluation_seed": args.fresh_evaluation_seed,
        "evaluation_wasserstein": "mean marginal W1 over independent fresh generator samples",
        "checkpoint_selecting_batch": "retained as an audit column only",
        "middle_run_rule": "middle of three runs ranked by mean fresh-sample marginal Wasserstein distance",
        "tcnn_family_representative": {
            "configuration": family_representative.configuration,
            "width": family_representative.width,
            "run": family_representative.run,
            "fresh_marginal_w1": family_representative.fresh_wasserstein_mean,
            "selection_rule": "lowest fresh marginal W1 among independently ranked width-level middle runs",
        },
        "representative_plot_rule": "middle-performance run only",
        "quantitative_summary_rule": "all completed runs",
        "qq_quantiles": "240 equally spaced probabilities from 0.0025 to 0.9975",
        "stylized_estimator": "same global-centered, finite-window-corrected estimator used by analyze_four_model_middle_run.py; median across complete 16-step windows, lags 1-4",
        "interval": "95% max-t simultaneous bootstrap interval across lags 1-4",
        "real_interval_resampling": "moving-block bootstrap of contiguous returns followed by window reconstruction",
        "generated_interval_resampling": "whole-window bootstrap",
        "interpretation": "TCNN widths 2, 4, 6, and 8 form one family; figure 08 is a bounded parameter-budget comparison, not a scaling frontier",
        "generated_files": sorted(path.name for path in output.iterdir()),
    }
    with (output / "analysis_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Analysis complete: {output}")


if __name__ == "__main__":
    main()
