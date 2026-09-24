"""Create presentation figures based on the strict median-Wasserstein run.

This is intentionally separate from ``analyze_four_model_performance.py`` and
writes to new per-model folders.  A representative run is available only when
all three configured runs completed training; the same selected run is then
used for every non-training diagnostic for that model/layer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import analyze_four_model_performance as source


OUTPUT_FOLDER = "performance_analysis"
LAGS = np.arange(1, 5)
LAYERS = source.LAYERS
RUNS = source.RUNS
WINDOW_LENGTH = source.WINDOW_LENGTH
COLORS = source.LAYER_COLORS
BOOTSTRAP_REPLICATES = 2000
# Four analysis-window lengths keep most reconstructed 16-day windows away
# from artificial joins between resampled moving blocks.
BOOTSTRAP_BLOCK_LENGTH = 4 * WINDOW_LENGTH
BOOTSTRAP_BATCH_SIZE = 100
CONFIDENCE_LEVEL = 0.95
REAL_INTERVAL_CACHE: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}


def strict_roles(
    metrics: dict[tuple[int, int], source.RunMetrics], layer: int
) -> dict[str, int] | None:
    """Return best/middle/worst only when all three runs are complete."""
    if not all((layer, run) in metrics for run in RUNS):
        return None
    ranked = sorted(RUNS, key=lambda run: metrics[(layer, run)].best_loss)
    return {"best": ranked[0], "middle": ranked[1], "worst": ranked[2]}


def representative_runs(
    metrics: dict[tuple[int, int], source.RunMetrics],
    generated: dict[tuple[int, int], np.ndarray],
) -> dict[int, int]:
    selected: dict[int, int] = {}
    for layer in LAYERS:
        roles = strict_roles(metrics, layer)
        if roles is not None and (layer, roles["middle"]) in generated:
            selected[layer] = roles["middle"]
    return selected


def compute_window_metrics(windows: np.ndarray) -> dict[str, np.ndarray]:
    """Compute one estimate per original 16-step window at lags 1--4."""
    windows = source.as_windows(windows, Path("<in-memory>"))
    global_raw = float(np.mean(windows))
    global_abs = float(np.mean(np.abs(windows)))
    global_sq = float(np.mean(windows**2))
    result = {
        name: np.full((len(windows), len(LAGS)), np.nan)
        for name in source.METRIC_LABELS
    }
    raw = windows - global_raw
    absolute = np.abs(windows) - global_abs
    squared = windows**2 - global_sq
    var_raw = np.sum(raw**2, axis=1)
    var_abs = np.sum(absolute**2, axis=1)
    var_sq = np.sum(squared**2, axis=1)
    for lag_index, lag in enumerate(LAGS):
        correction = WINDOW_LENGTH / (WINDOW_LENGTH - lag)
        abs_numerator = np.sum(absolute[:, :-lag] * absolute[:, lag:], axis=1)
        raw_numerator = np.sum(raw[:, :-lag] * raw[:, lag:], axis=1)
        leverage_numerator = np.sum(raw[:, :-lag] * squared[:, lag:], axis=1)
        np.divide(
            abs_numerator * correction,
            var_abs,
            out=result["abs_acf"][:, lag_index],
            where=var_abs > 0,
        )
        np.divide(
            raw_numerator * correction,
            var_raw,
            out=result["acf"][:, lag_index],
            where=var_raw > 0,
        )
        leverage_denominator = np.sqrt(var_raw * var_sq)
        np.divide(
            leverage_numerator * correction,
            leverage_denominator,
            out=result["leverage"][:, lag_index],
            where=leverage_denominator > 0,
        )
    return result


def simultaneous_interval(
    center: np.ndarray,
    bootstrap_estimates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct a max-t simultaneous interval across the four displayed lags."""
    center = np.asarray(center, dtype=float)
    estimates = np.asarray(bootstrap_estimates, dtype=float)
    standard_error = np.nanstd(estimates, axis=0, ddof=1)
    usable = np.isfinite(standard_error) & (standard_error > np.finfo(float).eps)
    lower = center.copy()
    upper = center.copy()
    if not np.any(usable):
        return lower, upper
    standardized = np.abs(
        (estimates[:, usable] - center[usable]) / standard_error[usable]
    )
    max_statistic = np.nanmax(standardized, axis=1)
    critical_value = float(
        np.nanquantile(max_statistic, CONFIDENCE_LEVEL, method="linear")
    )
    lower[usable] = center[usable] - critical_value * standard_error[usable]
    upper[usable] = center[usable] + critical_value * standard_error[usable]
    return lower, upper


def moving_block_resample(
    values: np.ndarray,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample a full series by concatenating sampled overlapping blocks."""
    values = np.asarray(values, dtype=float)
    block_length = min(block_length, len(values))
    n_blocks = int(np.ceil(len(values) / block_length))
    starts = rng.integers(0, len(values) - block_length + 1, size=n_blocks)
    sampled = np.concatenate(
        [values[start : start + block_length] for start in starts]
    )
    return sampled[: len(values)]


def bootstrap_real_intervals(
    real: np.ndarray,
    centers: dict[str, np.ndarray],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Moving-block bootstrap the return series, then reconstruct rolling windows."""
    cache_key = hashlib.sha256(np.asarray(real, dtype=np.float64).tobytes()).hexdigest()
    if cache_key in REAL_INTERVAL_CACHE:
        return REAL_INTERVAL_CACHE[cache_key]
    rng = np.random.default_rng(20260828)
    estimates = {
        name: np.empty((BOOTSTRAP_REPLICATES, len(LAGS)), dtype=float)
        for name in source.METRIC_LABELS
    }
    for replicate in range(BOOTSTRAP_REPLICATES):
        resampled = moving_block_resample(real, BOOTSTRAP_BLOCK_LENGTH, rng)
        resampled_windows = source.chop_windows(resampled, stride=1)
        replicate_metrics = compute_window_metrics(resampled_windows)
        for name in source.METRIC_LABELS:
            estimates[name][replicate] = np.nanmedian(
                replicate_metrics[name], axis=0
            )
    intervals = {
        name: simultaneous_interval(centers[name], estimates[name])
        for name in source.METRIC_LABELS
    }
    REAL_INTERVAL_CACHE[cache_key] = intervals
    return intervals


def bootstrap_generated_intervals(
    metrics: dict[str, np.ndarray],
    seed: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Bootstrap complete generated windows using common indices for all metrics."""
    centers = {
        name: np.nanmedian(values, axis=0) for name, values in metrics.items()
    }
    estimates = {
        name: np.empty((BOOTSTRAP_REPLICATES, len(LAGS)), dtype=float)
        for name in source.METRIC_LABELS
    }
    n_windows = len(next(iter(metrics.values())))
    rng = np.random.default_rng(seed)
    for start in range(0, BOOTSTRAP_REPLICATES, BOOTSTRAP_BATCH_SIZE):
        stop = min(start + BOOTSTRAP_BATCH_SIZE, BOOTSTRAP_REPLICATES)
        indices = rng.integers(
            0, n_windows, size=(stop - start, n_windows), dtype=np.int32
        )
        for name, values in metrics.items():
            estimates[name][start:stop] = np.nanmedian(values[indices], axis=1)
    return {
        name: simultaneous_interval(centers[name], estimates[name])
        for name in source.METRIC_LABELS
    }


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_distribution(
    model_name: str,
    real: np.ndarray,
    generated: dict[tuple[int, int], np.ndarray],
    selected: dict[int, int],
    output: Path,
) -> bool:
    if not selected:
        return False
    values = [real] + [generated[(layer, run)].reshape(-1) for layer, run in selected.items()]
    combined = np.concatenate(values)
    low, high = np.quantile(combined, [0.001, 0.999])
    pad = 0.05 * (high - low)
    grid = np.linspace(low - pad, high + pad, 700)

    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    ax.hist(
        real,
        bins=55,
        range=(grid[0], grid[-1]),
        density=True,
        color="#9b9ba2",
        alpha=0.62,
        label="Bitcoin",
    )
    for layer, run in selected.items():
        ax.plot(
            grid,
            source.density(generated[(layer, run)].reshape(-1), grid),
            color=COLORS[layer],
            linewidth=2.6,
            label=f"Layer {layer}",
        )
    ax.set(
        xlabel="Log return",
        ylabel="Density",
        title=f"{model_name}: return distribution by generator depth",
        xlim=(grid[0], grid[-1]),
    )
    ax.legend(frameon=True)
    fig.tight_layout()
    save(fig, output)
    return True


def plot_qq(
    model_name: str,
    real: np.ndarray,
    generated: dict[tuple[int, int], np.ndarray],
    selected: dict[int, int],
    output: Path,
) -> bool:
    if not selected:
        return False
    real_q = np.quantile(real, source.QUANTILES)
    curves = {
        layer: np.quantile(generated[(layer, run)].reshape(-1), source.QUANTILES)
        for layer, run in selected.items()
    }
    bound = 1.05 * max(
        float(np.max(np.abs(values))) for values in [real_q, *curves.values()]
    )
    fig, ax = plt.subplots(figsize=(7.6, 7.2))
    ax.plot(
        [-bound, bound],
        [-bound, bound],
        "--",
        color="#222222",
        linewidth=1.4,
        label="45° reference",
    )
    for layer, quantiles in curves.items():
        ax.plot(
            real_q,
            quantiles,
            color=COLORS[layer],
            linewidth=2.4,
            label=f"Layer {layer}",
        )
    ax.set(
        xlabel="Bitcoin quantile",
        ylabel="Generated quantile",
        title=f"{model_name}: Q–Q plot by generator depth",
        xlim=(-bound, bound),
        ylim=(-bound, bound),
    )
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=True)
    fig.tight_layout()
    save(fig, output)
    return True


def plot_dependence(
    model_name: str,
    metric_name: str,
    real_metrics: dict[str, np.ndarray],
    real_intervals: dict[str, tuple[np.ndarray, np.ndarray]],
    generated_metrics: dict[tuple[int, int], dict[str, np.ndarray]],
    generated_intervals: dict[
        tuple[int, int], dict[str, tuple[np.ndarray, np.ndarray]]
    ],
    selected: dict[int, int],
    output: Path,
) -> bool:
    usable = {
        layer: run
        for layer, run in selected.items()
        if (layer, run) in generated_metrics
    }
    if not usable:
        return False
    title, ylabel = source.METRIC_LABELS[metric_name]
    real_center = np.nanmedian(real_metrics[metric_name], axis=0)
    real_low, real_high = real_intervals[metric_name]

    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    ax.axhline(0, color="#111111", linewidth=1.8, alpha=0.82, zorder=1)
    ax.fill_between(
        LAGS,
        real_low,
        real_high,
        color="#333333",
        alpha=0.11,
        linewidth=0,
        zorder=2,
    )
    ax.plot(
        LAGS,
        real_center,
        "--",
        color="#222222",
        linewidth=2.3,
        label="Bitcoin",
        zorder=4,
    )
    for lag, center, low, high in zip(LAGS, real_center, real_low, real_high):
        contains_zero = low <= 0 <= high
        ax.scatter(
            lag,
            center,
            marker="D",
            s=42,
            facecolors="none" if contains_zero else "#222222",
            edgecolors="#222222",
            linewidths=1.5,
            zorder=5,
        )
    for layer, run in usable.items():
        center = np.nanmedian(generated_metrics[(layer, run)][metric_name], axis=0)
        low, high = generated_intervals[(layer, run)][metric_name]
        ax.fill_between(
            LAGS,
            low,
            high,
            color=COLORS[layer],
            alpha=0.12,
            linewidth=0,
            zorder=2,
        )
        ax.plot(
            LAGS,
            center,
            "-",
            color=COLORS[layer],
            linewidth=2.6,
            label=f"Layer {layer}",
            zorder=3,
        )
        for lag, estimate, interval_low, interval_high in zip(
            LAGS, center, low, high
        ):
            contains_zero = interval_low <= 0 <= interval_high
            ax.scatter(
                lag,
                estimate,
                marker="o",
                s=44,
                facecolors="none" if contains_zero else COLORS[layer],
                edgecolors=COLORS[layer],
                linewidths=1.5,
                zorder=5,
            )
    ax.set(
        xlabel="Lag",
        ylabel=ylabel,
        title=f"{model_name}: {title}",
        xticks=LAGS,
    )
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(
        Patch(
            facecolor="#777777",
            alpha=0.18,
            label="95% simultaneous bootstrap interval",
        )
    )
    labels.append("95% simultaneous bootstrap interval")
    ax.legend(handles, labels, frameon=True, ncol=2)
    fig.tight_layout()
    save(fig, output)
    return True


def plot_abs_acf_violin(
    model_name: str,
    real_metrics: dict[str, np.ndarray],
    generated_metrics: dict[tuple[int, int], dict[str, np.ndarray]],
    selected: dict[int, int],
    output: Path,
) -> bool:
    rows: list[tuple[str, np.ndarray | None, str]] = [
        ("Bitcoin", real_metrics["abs_acf"], "#707078")
    ]
    rows.extend(
        (
            f"Layer {layer}",
            generated_metrics[(layer, selected[layer])]["abs_acf"]
            if layer in selected and (layer, selected[layer]) in generated_metrics
            else None,
            COLORS[layer],
        )
        for layer in LAYERS
    )
    available_values = [values for _, values, _ in rows if values is not None]
    if len(available_values) == 1:
        return False
    finite = np.concatenate([values[np.isfinite(values)] for values in available_values])
    low, high = np.quantile(finite, [0.002, 0.998])
    pad = 0.06 * (high - low)

    fig, axes = plt.subplots(5, 1, figsize=(12.2, 12.0), sharex=True, sharey=True)
    for ax, (label, values, color) in zip(axes, rows):
        ax.axhline(0, color="#111111", linewidth=1.35, alpha=0.75, zorder=1)
        if values is None:
            ax.text(
                0.5,
                0.5,
                "Representative run unavailable",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#666666",
            )
        else:
            datasets = [values[:, index][np.isfinite(values[:, index])] for index in range(len(LAGS))]
            violin = ax.violinplot(
                datasets,
                positions=LAGS,
                widths=0.72,
                showmeans=False,
                showmedians=True,
                showextrema=False,
            )
            for body in violin["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.54)
                body.set_linewidth(0.8)
            violin["cmedians"].set_color(color)
            violin["cmedians"].set_linewidth(2.0)
        ax.set_ylabel("ACF")
        ax.set_title(label, loc="left", fontsize=11, fontweight="semibold")
        ax.set_ylim(low - pad, high + pad)
        ax.grid(True, axis="y", alpha=0.20)
    axes[-1].set_xlabel("Lag")
    axes[-1].set_xticks(LAGS)
    fig.suptitle(
        f"{model_name}: absolute-return ACF window distributions",
        fontsize=14.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, output)
    return True


def plot_wasserstein(
    model_name: str,
    metrics: dict[tuple[int, int], source.RunMetrics],
    output: Path,
) -> bool:
    if not any(item.losses is not None and len(item.losses) for item in metrics.values()):
        return False
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.4), sharex=True, sharey=True)
    plotted = False
    for layer, ax in zip(LAYERS, axes.flat):
        traces = [
            (run, metrics[(layer, run)])
            for run in RUNS
            if (layer, run) in metrics
            and metrics[(layer, run)].losses is not None
            and len(metrics[(layer, run)].losses)
        ]
        color = COLORS[layer]
        ax.axhline(0, color="#111111", linewidth=0.8, alpha=0.25)
        if not traces:
            ax.text(
                0.5,
                0.5,
                "No completed training history",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"Layer {layer}")
            continue
        for _, item in traces:
            ax.plot(
                item.epochs,
                item.losses,
                color=color,
                linewidth=0.8,
                alpha=0.15,
                zorder=1,
            )
        roles = strict_roles(metrics, layer)
        if roles is not None:
            starts = [float(item.epochs[0]) for _, item in traces]
            ends = [float(item.epochs[-1]) for _, item in traces]
            common_start, common_end = max(starts), min(ends)
            common_epochs = np.unique(
                np.concatenate(
                    [
                        item.epochs[
                            (item.epochs >= common_start) & (item.epochs <= common_end)
                        ]
                        for _, item in traces
                    ]
                )
            )
            smoothed = {
                run: source.ema(
                    np.interp(common_epochs, item.epochs, item.losses)
                )
                for run, item in traces
            }
            all_smoothed = np.vstack(list(smoothed.values()))
            ax.fill_between(
                common_epochs,
                np.min(all_smoothed, axis=0),
                np.max(all_smoothed, axis=0),
                color=color,
                alpha=0.18,
                linewidth=0,
                zorder=2,
            )
            ax.plot(
                common_epochs,
                smoothed[roles["middle"]],
                color=color,
                linewidth=3.1,
                zorder=4,
            )
        else:
            ax.text(
                0.98,
                0.96,
                "No three-run median",
                ha="right",
                va="top",
                transform=ax.transAxes,
                color="#666666",
                fontsize=9,
            )
        ax.set_title(f"Layer {layer}")
        ax.set_yscale("log")
        ax.grid(True, which="major", alpha=0.25)
        ax.grid(True, which="minor", axis="y", alpha=0.09)
        plotted = True
    if not plotted:
        plt.close(fig)
        return False
    fig.supxlabel("Epoch", y=0.105)
    fig.supylabel("Wasserstein loss (log scale)")
    fig.suptitle(f"{model_name}: Wasserstein training history", fontsize=15)
    handles = [
        Line2D(
            [0],
            [0],
            color="#4c78a8",
            linewidth=0.8,
            alpha=0.20,
            label="raw completed-run histories",
        ),
        Patch(
            facecolor="#4c78a8",
            alpha=0.18,
            label="pointwise across-seed EMA range",
        ),
        Line2D(
            [0],
            [0],
            color="#4c78a8",
            linewidth=3.1,
            label=f"middle-performance run, EMA span {source.EMA_SPAN}",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0.035, 0.16, 1, 0.95))
    save(fig, output)
    return True


def write_selection_table(
    output: Path,
    metrics: dict[tuple[int, int], source.RunMetrics],
    generated: dict[tuple[int, int], np.ndarray],
) -> None:
    rows: list[dict[str, object]] = []
    for layer in LAYERS:
        roles = strict_roles(metrics, layer)
        role_by_run = (
            {run: role for role, run in roles.items()} if roles is not None else {}
        )
        for run in RUNS:
            item = metrics.get((layer, run))
            rows.append(
                {
                    "layer": layer,
                    "run": run,
                    "strict_three_run_role": role_by_run.get(
                        run, "unranked/incomplete/missing"
                    ),
                    "used_for_primary_diagnostics": bool(
                        roles is not None
                        and run == roles["middle"]
                        and (layer, run) in generated
                    ),
                    "best_wasserstein": item.best_loss if item else np.nan,
                    "best_epoch": item.best_epoch if item else np.nan,
                    "generated_windows_available": (layer, run) in generated,
                    "n_generated_windows": (
                        len(generated[(layer, run)]) if (layer, run) in generated else 0
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(output, index=False)


def analyze_model(
    model_name: str,
    sweep: Path,
    wasserstein_filename: str = "07_wasserstein_loss_vs_epoch.png",
    wasserstein_plotter: Callable[
        [str, dict[tuple[int, int], source.RunMetrics], Path], bool
    ] = plot_wasserstein,
) -> dict[str, object]:
    output = sweep.parent / OUTPUT_FOLDER
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "model": model_name,
        "sweep": sweep.relative_to(source.ROOT).as_posix(),
        "output": output.relative_to(source.ROOT).as_posix(),
        "selection_rule": (
            "For each layer with three completed runs, select the run with the "
            "median best Wasserstein loss; use it for all non-training diagnostics."
        ),
        "lags": LAGS.tolist(),
        "confidence_intervals": {
            "level": CONFIDENCE_LEVEL,
            "type": "max-t simultaneous bootstrap interval across displayed lags",
            "replicates": BOOTSTRAP_REPLICATES,
            "bitcoin_method": (
                "moving-block bootstrap of the original return series, followed "
                "by reconstruction of stride-1 rolling 16-step windows"
            ),
            "bitcoin_block_length": BOOTSTRAP_BLOCK_LENGTH,
            "bitcoin_block_length_rationale": (
                "four analysis-window lengths, limiting reconstructed windows "
                "that cross artificial joins between resampled blocks"
            ),
            "generated_method": "bootstrap resampling of complete 16-step windows",
        },
        "generated_files": [],
        "notes": [],
    }
    if not sweep.exists():
        manifest["notes"].append("Configured sweep folder does not exist.")
        (output / "analysis_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return manifest

    generated, generated_notes = source.load_generated_windows(sweep)
    metrics, metric_notes = source.load_run_metrics(sweep)
    generated = {key: windows for key, windows in generated.items() if key in metrics}
    selected = representative_runs(metrics, generated)
    manifest["notes"].extend(generated_notes + metric_notes)
    for layer in LAYERS:
        if strict_roles(metrics, layer) is None:
            manifest["notes"].append(
                f"Layer {layer} omitted from representative diagnostics: "
                "three completed runs are not available."
            )
        elif layer not in selected:
            manifest["notes"].append(
                f"Layer {layer} omitted from representative diagnostics: "
                "selected run has no generated-window cache."
            )

    real = source.load_real_returns(sweep.parent)
    real_windows = source.chop_windows(real, stride=1)
    real_metrics = compute_window_metrics(real_windows)
    selected_generated = {
        (layer, run): generated[(layer, run)] for layer, run in selected.items()
    }
    generated_metrics = {
        key: compute_window_metrics(windows)
        for key, windows in selected_generated.items()
    }
    real_centers = {
        name: np.nanmedian(values, axis=0) for name, values in real_metrics.items()
    }
    real_intervals = bootstrap_real_intervals(real, real_centers)
    generated_intervals = {
        (layer, run): bootstrap_generated_intervals(
            generated_metrics[(layer, run)],
            seed=20260828 + 100 * layer + run,
        )
        for layer, run in selected.items()
    }
    write_selection_table(
        output / "run_selection_and_availability.csv", metrics, generated
    )

    tasks = [
        (
            "01_return_distribution.png",
            lambda path: plot_distribution(
                model_name, real, generated, selected, path
            ),
        ),
        (
            "02_qq_plot.png",
            lambda path: plot_qq(model_name, real, generated, selected, path),
        ),
        (
            "03_absolute_acf.png",
            lambda path: plot_dependence(
                model_name,
                "abs_acf",
                real_metrics,
                real_intervals,
                generated_metrics,
                generated_intervals,
                selected,
                path,
            ),
        ),
        (
            "04_absolute_acf_violin.png",
            lambda path: plot_abs_acf_violin(
                model_name, real_metrics, generated_metrics, selected, path
            ),
        ),
        (
            "05_acf.png",
            lambda path: plot_dependence(
                model_name,
                "acf",
                real_metrics,
                real_intervals,
                generated_metrics,
                generated_intervals,
                selected,
                path,
            ),
        ),
        (
            "06_leverage.png",
            lambda path: plot_dependence(
                model_name,
                "leverage",
                real_metrics,
                real_intervals,
                generated_metrics,
                generated_intervals,
                selected,
                path,
            ),
        ),
        (
            wasserstein_filename,
            lambda path: wasserstein_plotter(model_name, metrics, path),
        ),
    ]
    for filename, task in tasks:
        try:
            if task(output / filename):
                manifest["generated_files"].append(filename)
            else:
                manifest["notes"].append(
                    f"Skipped {filename}: required completed-run data unavailable."
                )
        except Exception as exc:
            manifest["notes"].append(
                f"Failed {filename}: {type(exc).__name__}: {exc}"
            )

    manifest["representative_runs"] = {
        f"layer_{layer}": f"run_{run}" for layer, run in selected.items()
    }
    manifest["available_metric_runs"] = [
        f"layer_{layer}/run_{run}" for layer, run in sorted(metrics)
    ]
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    source.configure_plotting()
    summaries = [
        analyze_model(model_name, sweep)
        for model_name, sweep in source.SWEEPS.items()
    ]
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
