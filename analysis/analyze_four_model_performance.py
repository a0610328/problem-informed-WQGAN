"""Generate per-model performance figures from saved sweep artifacts.

Only the four sweep folders listed in SWEEPS are inspected. Generated-data
figures use saved 16-step windows only; this script never invokes a model or
concatenates windows before estimating serial-dependence statistics.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parent.parent
SWEEPS = {
    "ZZ": ROOT / "project_model_ZZ" / "sweep_btc_daily_2020_2026_20260813_105114",
    "ZZ-CRX": ROOT / "project_model_ZZ_CRZ" / "sweep_btc_daily_2020_2026_20260827_112027",
    "ZZ time evolution": ROOT
    / "project_model_ZZ_time_evolution"
    / "sweep_btc_daily_2020_2026_20260814_185721",
    "HEA": ROOT / "baseline_model_HEA" / "sweep_btc_daily_2020_2026_20260802_215447",
}
OUTPUT_FOLDER = "performance_analysis"
DATASET = "btc_daily_2020_2026.csv"
LAYERS = (1, 2, 3, 4)
RUNS = (0, 1, 2)
LAGS = np.arange(1, 7)
WINDOW_LENGTH = 16
QUANTILES = np.linspace(0.0025, 0.9975, 240)
EMA_SPAN = 41
LAYER_COLORS = {
    1: "#1f77b4",
    2: "#ff7f0e",
    3: "#2ca02c",
    4: "#9467bd",
}
METRIC_LABELS = {
    "abs_acf": ("Absolute-return ACF", "ACF"),
    "acf": ("Return ACF", "ACF"),
    "leverage": ("Leverage correlation", "Correlation"),
}
LAYER_RE = re.compile(r"(?:^|[\\/])layers_(\d+)(?:_|[\\/])")
RUN_RE = re.compile(r"run_(\d+)")


@dataclass
class RunMetrics:
    best_loss: float
    best_epoch: int
    epochs: np.ndarray | None
    losses: np.ndarray | None


def configure_plotting() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "sans-serif",
            "font.size": 10.5,
        }
    )


def scalar(value: np.ndarray) -> object:
    return np.asarray(value).reshape(-1)[0].item()


def parse_layer_run(path: Path, npz: np.lib.npyio.NpzFile | None = None) -> tuple[int | None, int | None]:
    layer = int(scalar(npz["layer"])) if npz is not None and "layer" in npz.files else None
    run = int(scalar(npz["run_id"])) if npz is not None and "run_id" in npz.files else None
    if layer is None:
        match = LAYER_RE.search(str(path))
        layer = int(match.group(1)) if match else None
    if run is None:
        match = RUN_RE.search(str(path))
        run = int(match.group(1)) if match else None
    return layer, run


def as_windows(values: np.ndarray, source: Path) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        if array.size % WINDOW_LENGTH:
            raise ValueError(f"Cannot reshape {array.size} values into 16-step windows: {source}")
        array = array.reshape(-1, WINDOW_LENGTH)
    elif array.ndim > 2:
        array = array.reshape(-1, array.shape[-1])
    if array.ndim != 2 or array.shape[1] != WINDOW_LENGTH:
        raise ValueError(f"Expected (*, 16) windows, found {array.shape}: {source}")
    finite_rows = np.all(np.isfinite(array), axis=1)
    return array[finite_rows]


def load_generated_windows(sweep: Path) -> tuple[dict[tuple[int, int], np.ndarray], list[str]]:
    found: dict[tuple[int, int], np.ndarray] = {}
    notes: list[str] = []
    for path in sorted(sweep.rglob("*.npz")):
        if "generated" not in path.name.lower() and "generated_cache" not in str(path).lower():
            continue
        try:
            with np.load(path, allow_pickle=False) as npz:
                data_key = next(
                    (key for key in ("generated_windows", "generated_log_returns", "generated_returns") if key in npz.files),
                    None,
                )
                if data_key is None:
                    continue
                layer, run = parse_layer_run(path, npz)
                if layer not in LAYERS or run not in RUNS:
                    notes.append(f"Ignored unidentifiable cache: {path}")
                    continue
                if "dataset_file" in npz.files:
                    dataset = scalar(npz["dataset_file"])
                    if isinstance(dataset, bytes):
                        dataset = dataset.decode("utf-8")
                    if Path(str(dataset)).name != DATASET:
                        notes.append(f"Ignored cache for {dataset}: {path}")
                        continue
                windows = as_windows(npz[data_key], path)
        except Exception as exc:
            notes.append(f"Could not read {path}: {exc}")
            continue
        key = (layer, run)
        if key in found:
            notes.append(f"Ignored duplicate generated cache for layer {layer}, run {run}: {path}")
        elif len(windows):
            found[key] = windows

    # CSV is a fallback only when an NPZ cache for that layer/run was not found.
    for path in sorted(sweep.rglob("generated_windows*.csv")):
        layer, run = parse_layer_run(path)
        if layer not in LAYERS or run not in RUNS or (layer, run) in found:
            continue
        try:
            frame = pd.read_csv(path)
            frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed")]
            numeric = frame.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
            windows = as_windows(numeric.to_numpy(), path)
            if len(windows):
                found[(layer, run)] = windows
        except Exception as exc:
            notes.append(f"Could not read {path}: {exc}")
    return found, notes


def load_run_metrics(sweep: Path) -> tuple[dict[tuple[int, int], RunMetrics], list[str]]:
    found: dict[tuple[int, int], RunMetrics] = {}
    notes: list[str] = []
    for layer_folder in sorted(sweep.glob("layers_*")):
        layer, _ = parse_layer_run(layer_folder)
        if not layer_folder.is_dir() or layer not in LAYERS:
            continue
        for run_folder in sorted(layer_folder.glob("run_*")):
            _, run = parse_layer_run(run_folder)
            if not run_folder.is_dir() or run not in RUNS:
                continue
            metrics_folder = run_folder / "metrics"
            best_path = metrics_folder / "best_metrics.txt"
            if not best_path.exists():
                notes.append(f"Missing best_metrics.txt: layer {layer}, run {run}")
                continue
            try:
                epoch_path = metrics_folder / "loss_epochs.txt"
                last_metric_epoch = None
                if epoch_path.exists():
                    recorded_epochs = np.atleast_1d(np.loadtxt(epoch_path, dtype=float))
                    finite_epochs = recorded_epochs[np.isfinite(recorded_epochs)]
                    if len(finite_epochs):
                        last_metric_epoch = float(finite_epochs[-1])
                qq_epochs = []
                for qq_path in (run_folder / "plots").glob("QQ_plot_epoch_*.pdf"):
                    match = re.search(r"QQ_plot_epoch_(\d+)$", qq_path.stem)
                    if match:
                        qq_epochs.append(int(match.group(1)))
                max_qq_epoch = max(qq_epochs) if qq_epochs else None
                if not (
                    (last_metric_epoch is not None and last_metric_epoch >= 4000)
                    or (max_qq_epoch is not None and max_qq_epoch >= 4000)
                ):
                    notes.append(
                        f"Ignored incomplete training: layer {layer}, run {run} "
                        f"(last metric={last_metric_epoch}, max QQ={max_qq_epoch})"
                    )
                    continue
                best = np.atleast_1d(np.loadtxt(best_path, dtype=float))
                if len(best) < 2 or not np.isfinite(best[:2]).all():
                    raise ValueError("expected two finite numbers")
                loss_path = metrics_folder / "loss_wass.txt"
                epochs = losses = None
                if epoch_path.exists() and loss_path.exists():
                    epochs = np.atleast_1d(np.loadtxt(epoch_path, dtype=float))
                    losses = np.atleast_1d(np.loadtxt(loss_path, dtype=float))
                    length = min(len(epochs), len(losses))
                    epochs, losses = epochs[:length], losses[:length]
                    valid = np.isfinite(epochs) & np.isfinite(losses) & (losses > 0)
                    epochs, losses = epochs[valid], losses[valid]
                found[(layer, run)] = RunMetrics(float(best[0]), int(round(best[1])), epochs, losses)
            except Exception as exc:
                notes.append(f"Could not read metrics for layer {layer}, run {run}: {exc}")
    return found, notes


def run_roles(metrics: dict[tuple[int, int], RunMetrics], layer: int) -> dict[str, int | None]:
    ranked = sorted(
        (run for run in RUNS if (layer, run) in metrics),
        key=lambda run: metrics[(layer, run)].best_loss,
    )
    if not ranked:
        return {"best": None, "middle": None, "worst": None}
    # A named middle-performing run exists only with an odd number of runs.
    middle = ranked[len(ranked) // 2] if len(ranked) % 2 == 1 else None
    return {"best": ranked[0], "middle": middle, "worst": ranked[-1]}


def load_real_returns(model_root: Path) -> np.ndarray:
    path = ROOT / "Bitcoin_Data_2020_2026" / DATASET
    frame = pd.read_csv(path)
    if "Log_return" not in frame.columns:
        raise KeyError(f"Log_return column missing from {path}")
    values = pd.to_numeric(frame["Log_return"], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError(f"No finite real returns in {path}")
    return values


def chop_windows(values: np.ndarray, stride: int = 1) -> np.ndarray:
    if len(values) < WINDOW_LENGTH:
        return np.empty((0, WINDOW_LENGTH))
    starts = np.arange(0, len(values) - WINDOW_LENGTH + 1, stride)
    return np.stack([values[start : start + WINDOW_LENGTH] for start in starts])


def compute_window_metrics(windows: np.ndarray) -> dict[str, np.ndarray]:
    """Return one serial-dependence estimate per original 16-step window and lag."""
    windows = as_windows(windows, Path("<in-memory>"))
    global_raw = float(np.mean(windows))
    global_abs = float(np.mean(np.abs(windows)))
    global_sq = float(np.mean(windows**2))
    result = {name: np.full((len(windows), len(LAGS)), np.nan) for name in METRIC_LABELS}
    for row_index, window in enumerate(windows):
        raw = window - global_raw
        absolute = np.abs(window) - global_abs
        squared = window**2 - global_sq
        var_raw = np.sum(raw**2)
        var_abs = np.sum(absolute**2)
        var_sq = np.sum(squared**2)
        for lag_index, lag in enumerate(LAGS):
            correction = WINDOW_LENGTH / (WINDOW_LENGTH - lag)
            if var_abs > 0:
                result["abs_acf"][row_index, lag_index] = (
                    np.sum(absolute[:-lag] * absolute[lag:]) / var_abs * correction
                )
            if var_raw > 0:
                result["acf"][row_index, lag_index] = (
                    np.sum(raw[:-lag] * raw[lag:]) / var_raw * correction
                )
            if var_raw > 0 and var_sq > 0:
                result["leverage"][row_index, lag_index] = (
                    np.sum(raw[:-lag] * squared[lag:]) / np.sqrt(var_raw * var_sq) * correction
                )
    return result


def save_figure(fig: plt.Figure, output: Path) -> None:
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def density(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2 or np.std(values) == 0:
        return np.zeros_like(grid)
    return gaussian_kde(values)(grid)


def plot_distribution(
    model_name: str,
    real: np.ndarray,
    generated: dict[tuple[int, int], np.ndarray],
    metrics: dict[tuple[int, int], RunMetrics],
    output: Path,
) -> bool:
    usable: list[tuple[int, dict[str, int | None]]] = []
    all_values = [real]
    for layer in LAYERS:
        roles = run_roles(metrics, layer)
        middle = roles["middle"]
        if middle is not None and (layer, middle) in generated:
            usable.append((layer, roles))
            all_values.extend(generated[(layer, run)].reshape(-1) for run in RUNS if (layer, run) in generated)
    if not usable:
        return False
    combined = np.concatenate(all_values)
    low, high = np.quantile(combined, [0.001, 0.999])
    pad = 0.05 * (high - low)
    grid = np.linspace(low - pad, high + pad, 700)
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    ax.hist(real, bins=55, range=(grid[0], grid[-1]), density=True, color="#9b9ba2", alpha=0.62, label="Bitcoin")
    for layer, roles in usable:
        color = LAYER_COLORS[layer]
        middle = int(roles["middle"])
        ax.plot(grid, density(generated[(layer, middle)].reshape(-1), grid), color=color, linewidth=2.6, label=f"Layer {layer} (middle run {middle})")
        boundary_curves = [
            density(generated[(layer, int(roles[role]))].reshape(-1), grid)
            for role in ("best", "worst")
            if roles[role] is not None and (layer, int(roles[role])) in generated
        ]
        if len(boundary_curves) == 2:
            ax.fill_between(grid, np.minimum(*boundary_curves), np.maximum(*boundary_curves), color=color, alpha=0.17, linewidth=0)
    ax.set(xlabel="Log return", ylabel="Density", title=f"{model_name}: return distribution by generator depth")
    ax.set_xlim(grid[0], grid[-1])
    ax.legend(frameon=True)
    fig.tight_layout()
    save_figure(fig, output)
    return True


def plot_qq(
    model_name: str,
    real: np.ndarray,
    generated: dict[tuple[int, int], np.ndarray],
    metrics: dict[tuple[int, int], RunMetrics],
    output: Path,
) -> bool:
    real_q = np.quantile(real, QUANTILES)
    curves: list[np.ndarray] = []
    items: list[tuple[int, dict[str, int | None], np.ndarray]] = []
    for layer in LAYERS:
        roles = run_roles(metrics, layer)
        middle = roles["middle"]
        if middle is None or (layer, middle) not in generated:
            continue
        middle_q = np.quantile(generated[(layer, middle)].reshape(-1), QUANTILES)
        items.append((layer, roles, middle_q))
        curves.append(middle_q)
    if not items:
        return False
    bound = 1.05 * max(float(np.max(np.abs(values))) for values in [real_q, *curves])
    fig, ax = plt.subplots(figsize=(7.6, 7.2))
    ax.plot([-bound, bound], [-bound, bound], "--", color="#222222", linewidth=1.4, label="45° reference")
    for layer, roles, middle_q in items:
        color = LAYER_COLORS[layer]
        middle = int(roles["middle"])
        ax.plot(real_q, middle_q, color=color, linewidth=2.4, label=f"Layer {layer} (middle run {middle})")
        bounds = [
            np.quantile(generated[(layer, int(roles[role]))].reshape(-1), QUANTILES)
            for role in ("best", "worst")
            if roles[role] is not None and (layer, int(roles[role])) in generated
        ]
        if len(bounds) == 2:
            ax.fill_between(real_q, np.minimum(*bounds), np.maximum(*bounds), color=color, alpha=0.17, linewidth=0)
    ax.set(xlabel="Bitcoin quantile", ylabel="Generated quantile", title=f"{model_name}: Q–Q plot by generator depth")
    ax.set_xlim(-bound, bound)
    ax.set_ylim(-bound, bound)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=True)
    fig.tight_layout()
    save_figure(fig, output)
    return True


def plot_dependence_summary(
    model_name: str,
    metric_name: str,
    real_metric: np.ndarray,
    generated_metrics: dict[tuple[int, int], dict[str, np.ndarray]],
    output: Path,
) -> bool:
    if not any(key[0] in LAYERS for key in generated_metrics):
        return False
    title, ylabel = METRIC_LABELS[metric_name]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2), sharex=True, sharey=True)
    real_center = np.nanmedian(real_metric, axis=0)
    real_low, real_high = np.nanpercentile(real_metric, [25, 75], axis=0)
    plotted = False
    for layer, ax in zip(LAYERS, axes.flat):
        layer_runs = [(run, generated_metrics[(layer, run)][metric_name]) for run in RUNS if (layer, run) in generated_metrics]
        ax.fill_between(LAGS, real_low, real_high, color="#666666", alpha=0.12, linewidth=0)
        ax.plot(LAGS, real_center, "--o", color="#222222", linewidth=2.0, markersize=4, label="Bitcoin median (IQR)")
        if not layer_runs:
            ax.text(0.5, 0.5, "No generated-window data", ha="center", va="center", transform=ax.transAxes, color="#666666")
        else:
            seed_centers = []
            for run, values in layer_runs:
                center = np.nanmedian(values, axis=0)
                seed_centers.append(center)
                ax.plot(LAGS, center, color=LAYER_COLORS[layer], linewidth=1.0, alpha=0.38, marker="o", markersize=2.8, label=f"run {run}")
            seed_centers_array = np.asarray(seed_centers)
            across_center = np.nanmedian(seed_centers_array, axis=0)
            ax.fill_between(LAGS, np.nanmin(seed_centers_array, axis=0), np.nanmax(seed_centers_array, axis=0), color=LAYER_COLORS[layer], alpha=0.20, linewidth=0, label="seed min–max")
            ax.plot(LAGS, across_center, color=LAYER_COLORS[layer], linewidth=3.0, marker="o", markersize=4.5, label="across-seed median")
            plotted = True
        ax.axhline(0, color="#777777", linewidth=0.8, alpha=0.5)
        ax.set_title(f"Layer {layer}")
        ax.set_xticks(LAGS)
    if not plotted:
        plt.close(fig)
        return False
    fig.supxlabel("Lag", y=0.10)
    fig.supylabel(ylabel)
    fig.suptitle(f"{model_name}: {title} across saved seeds", fontsize=15)
    handles = [
        Line2D([0], [0], color="#222222", linestyle="--", marker="o", label="Bitcoin window median; IQR band"),
        Line2D([0], [0], color="#4c78a8", linewidth=1.0, alpha=0.45, marker="o", label="seed window median"),
        Line2D([0], [0], color="#4c78a8", linewidth=3.0, marker="o", label="across-seed median"),
        Patch(facecolor="#4c78a8", alpha=0.20, label="seed min–max"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.005), fontsize=9.5)
    fig.tight_layout(rect=(0.035, 0.15, 1, 0.95))
    save_figure(fig, output)
    return True


def plot_abs_acf_violin(
    model_name: str,
    layer: int,
    real_abs_acf: np.ndarray,
    generated_metrics: dict[tuple[int, int], dict[str, np.ndarray]],
    output: Path,
) -> bool:
    available = [run for run in RUNS if (layer, run) in generated_metrics]
    if not available:
        return False
    rows = [("Bitcoin", real_abs_acf, "Greys")] + [
        (f"Generated seed/run {run}", generated_metrics[(layer, run)]["abs_acf"], "YlOrBr") for run in available
    ]
    fig, axes = plt.subplots(len(rows), 1, figsize=(12.2, 2.45 * len(rows) + 0.7), sharex=True, sharey=True, squeeze=False)
    columns = [f"Lag {lag}" for lag in LAGS]
    for ax, (label, values, palette) in zip(axes[:, 0], rows):
        sns.violinplot(data=pd.DataFrame(values, columns=columns), ax=ax, palette=palette, inner="quartile", cut=0, linewidth=0.8)
        ax.axhline(0, color="#333333", linestyle="--", linewidth=0.9, alpha=0.55)
        ax.set_ylabel("ACF")
        ax.set_title(label, loc="left", fontsize=11, fontweight="semibold")
    axes[-1, 0].set_xlabel("Lag")
    fig.suptitle(f"{model_name}, layer {layer}: absolute-return ACF window distributions", fontsize=14.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, output)
    return True


def ema(values: np.ndarray, span: int = EMA_SPAN) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(values)
    out[0] = values[0]
    for index in range(1, len(values)):
        out[index] = alpha * values[index] + (1.0 - alpha) * out[index - 1]
    return out


def plot_wasserstein(
    model_name: str,
    metrics: dict[tuple[int, int], RunMetrics],
    output: Path,
) -> bool:
    if not any(item.losses is not None and len(item.losses) for item in metrics.values()):
        return False
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.4), sharex=True, sharey=True)
    plotted = False
    for layer, ax in zip(LAYERS, axes.flat):
        traces = [(run, metrics[(layer, run)]) for run in RUNS if (layer, run) in metrics and metrics[(layer, run)].losses is not None and len(metrics[(layer, run)].losses)]
        if not traces:
            ax.text(0.5, 0.5, "No Wasserstein history", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Layer {layer}")
            continue
        color = LAYER_COLORS[layer]
        for run, item in traces:
            ax.plot(item.epochs, item.losses, color=color, linewidth=0.75, alpha=0.15, zorder=1)
        roles = run_roles(metrics, layer)
        starts = [float(item.epochs[0]) for _, item in traces]
        ends = [float(item.epochs[-1]) for _, item in traces]
        common_start, common_end = max(starts), min(ends)
        common_epochs = np.unique(np.concatenate([item.epochs[(item.epochs >= common_start) & (item.epochs <= common_end)] for _, item in traces]))
        interpolated = np.vstack([np.interp(common_epochs, item.epochs, item.losses) for _, item in traces])
        mean_smoothed = ema(np.mean(interpolated, axis=0))
        role_traces: dict[str, np.ndarray] = {}
        for role in ("best", "middle", "worst"):
            role_run = roles[role]
            match = next((item for run, item in traces if run == role_run), None)
            if match is not None:
                role_traces[role] = ema(np.interp(common_epochs, match.epochs, match.losses))
        if "best" in role_traces and "worst" in role_traces and roles["best"] != roles["worst"]:
            ax.fill_between(common_epochs, np.minimum(role_traces["best"], role_traces["worst"]), np.maximum(role_traces["best"], role_traces["worst"]), color=color, alpha=0.18, linewidth=0, zorder=2)
        if "middle" in role_traces:
            ax.plot(common_epochs, role_traces["middle"], color="#303030", linewidth=1.35, alpha=0.8, zorder=3)
        ax.plot(common_epochs, mean_smoothed, color=color, linewidth=3.0, zorder=4)
        ax.set_title(f"Layer {layer} ({len(traces)} saved runs)")
        ax.set_yscale("log")
        ax.grid(True, which="major", alpha=0.25)
        ax.grid(True, which="minor", axis="y", alpha=0.09)
        plotted = True
    if not plotted:
        plt.close(fig)
        return False
    fig.supxlabel("Epoch", y=0.105)
    fig.supylabel("Wasserstein loss (log scale)")
    fig.suptitle(f"{model_name}: Wasserstein training history with ghosted seeds", fontsize=15)
    handles = [
        Line2D([0], [0], color="#4c78a8", linewidth=0.8, alpha=0.20, label="raw seed histories"),
        Patch(facecolor="#4c78a8", alpha=0.18, label="best–worst smoothed envelope"),
        Line2D([0], [0], color="#303030", linewidth=1.35, label="middle-performance run, EMA"),
        Line2D([0], [0], color="#4c78a8", linewidth=3.0, label=f"mean across available seeds, EMA span {EMA_SPAN}"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.005), fontsize=9.5)
    fig.tight_layout(rect=(0.035, 0.17, 1, 0.95))
    save_figure(fig, output)
    return True


def write_selection_table(
    output: Path,
    metrics: dict[tuple[int, int], RunMetrics],
    generated: dict[tuple[int, int], np.ndarray],
) -> None:
    rows = []
    for layer in LAYERS:
        roles = run_roles(metrics, layer)
        role_by_run = {
            run: "/".join(role for role, role_run in roles.items() if role_run == run)
            for run in RUNS
            if any(role_run == run for role_run in roles.values())
        }
        for run in RUNS:
            item = metrics.get((layer, run))
            rows.append(
                {
                    "layer": layer,
                    "run": run,
                    "performance_role": role_by_run.get(run, "unranked/missing"),
                    "best_wasserstein": item.best_loss if item else np.nan,
                    "best_epoch": item.best_epoch if item else np.nan,
                    "generated_windows_available": (layer, run) in generated,
                    "n_generated_windows": len(generated[(layer, run)]) if (layer, run) in generated else 0,
                }
            )
    pd.DataFrame(rows).to_csv(output, index=False)


def analyze_model(model_name: str, sweep: Path) -> dict[str, object]:
    model_root = sweep.parent
    output = model_root / OUTPUT_FOLDER
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "model": model_name,
        "sweep": sweep.relative_to(ROOT).as_posix(),
        "output": output.relative_to(ROOT).as_posix(),
        "generated_files": [],
        "notes": [],
    }
    if not sweep.exists():
        manifest["notes"].append("Configured sweep folder does not exist.")
        (output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    generated, generated_notes = load_generated_windows(sweep)
    metrics, metric_notes = load_run_metrics(sweep)
    generated = {key: windows for key, windows in generated.items() if key in metrics}
    manifest["notes"].extend(generated_notes + metric_notes)
    real = load_real_returns(model_root)
    real_windows = chop_windows(real, stride=1)
    real_metrics = compute_window_metrics(real_windows)
    generated_metrics = {key: compute_window_metrics(windows) for key, windows in generated.items()}
    write_selection_table(output / "run_selection_and_availability.csv", metrics, generated)

    tasks = [
        ("01_return_distribution.png", lambda path: plot_distribution(model_name, real, generated, metrics, path)),
        ("02_qq_plot.png", lambda path: plot_qq(model_name, real, generated, metrics, path)),
        ("03_absolute_acf.png", lambda path: plot_dependence_summary(model_name, "abs_acf", real_metrics["abs_acf"], generated_metrics, path)),
    ]
    for layer in LAYERS:
        tasks.append(
            (
                f"{3 + layer:02d}_absolute_acf_violin_layer_{layer}.png",
                lambda path, selected_layer=layer: plot_abs_acf_violin(model_name, selected_layer, real_metrics["abs_acf"], generated_metrics, path),
            )
        )
    tasks.extend(
        [
            ("08_acf.png", lambda path: plot_dependence_summary(model_name, "acf", real_metrics["acf"], generated_metrics, path)),
            ("09_leverage.png", lambda path: plot_dependence_summary(model_name, "leverage", real_metrics["leverage"], generated_metrics, path)),
            ("10_wasserstein_loss_vs_epoch.png", lambda path: plot_wasserstein(model_name, metrics, path)),
        ]
    )
    for filename, task in tasks:
        path = output / filename
        try:
            if task(path):
                manifest["generated_files"].append(filename)
            else:
                manifest["notes"].append(f"Skipped {filename}: required saved data unavailable.")
        except Exception as exc:
            manifest["notes"].append(f"Failed {filename}: {type(exc).__name__}: {exc}")

    manifest["available_generated_runs"] = [f"layer_{layer}/run_{run}" for layer, run in sorted(generated)]
    manifest["available_metric_runs"] = [f"layer_{layer}/run_{run}" for layer, run in sorted(metrics)]
    (output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    configure_plotting()
    summaries = [analyze_model(name, sweep) for name, sweep in SWEEPS.items()]
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
