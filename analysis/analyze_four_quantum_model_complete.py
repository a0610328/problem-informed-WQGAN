"""Reconstruct the finalized four-model figures from all completed runs.

This script uses all three runs for ranking and training-history presentation,
and uses the median-best-Wasserstein run for all distributional and
stylized-fact figures.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parent
DATA_PROJECT_ROOT = ROOT.parent
BITCOIN_DATA_ROOT = DATA_PROJECT_ROOT / "Bitcoin_Data_2020_2026"

# Load the helpers copied beside this script even when imported from elsewhere.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import analyze_four_model_middle_run as presentation
import analyze_four_model_performance as source
import generate_wasserstein_histories_with_marker as marked_history

_load_real_returns = source.load_real_returns


def load_real_returns(_model_root: Path) -> np.ndarray:
    """Read the shared 2020–2026 BTC data for every model."""
    return _load_real_returns(BITCOIN_DATA_ROOT)


def plot_wasserstein_with_marker(
    model_name: str,
    metrics: dict[tuple[int, int], source.RunMetrics],
    output: Path,
) -> bool:
    marked_history.plot_model(model_name, metrics, output)
    return True


OUTPUT_FOLDER = "performance_analysis"
CACHE_FOLDER_NAME = "performance_analysis_cache_complete_20260829"
COMPARISON_FOLDER = ROOT
HEA_OVERRIDE = ("baseline_model_HEA", 4, 0)


def _training_endpoint(run_folder: Path) -> tuple[float | None, int | None]:
    metric_epoch = None
    epoch_path = run_folder / "metrics" / "loss_epochs.txt"
    if epoch_path.exists():
        values = np.atleast_1d(np.loadtxt(epoch_path, dtype=float))
        values = values[np.isfinite(values)]
        if len(values):
            metric_epoch = float(values[-1])
    qq_epochs: list[int] = []
    for path in (run_folder / "plots").glob("QQ_plot_epoch_*.pdf"):
        match = re.search(r"QQ_plot_epoch_(\d+)$", path.stem)
        if match:
            qq_epochs.append(int(match.group(1)))
    return metric_epoch, max(qq_epochs) if qq_epochs else None


def load_run_metrics_complete(
    sweep: Path,
) -> tuple[dict[tuple[int, int], source.RunMetrics], list[str]]:
    found: dict[tuple[int, int], source.RunMetrics] = {}
    notes: list[str] = []
    for layer in source.LAYERS:
        layer_folder = sweep / f"layers_{layer}_qubit_8_stride_1"
        for run in source.RUNS:
            run_folder = layer_folder / f"run_{run}"
            metrics_folder = run_folder / "metrics"
            best_path = metrics_folder / "best_metrics.txt"
            # The compact public repository retains the selected generator and
            # reconstructed windows, but not the unused discriminator state.
            generator = run_folder / "weights" / "lowest_wass_generator.pkl"
            if not best_path.exists() or not generator.exists():
                notes.append(f"Missing required analysis artifacts: layer {layer}, run {run}")
                continue
            try:
                best = np.atleast_1d(np.loadtxt(best_path, dtype=float))
                if len(best) < 2 or not np.isfinite(best[:2]).all():
                    raise ValueError("expected two finite best-metric values")
                metric_epoch, qq_epoch = _training_endpoint(run_folder)
                reaches_4000 = bool(
                    (metric_epoch is not None and metric_epoch >= 4000)
                    or (qq_epoch is not None and qq_epoch >= 4000)
                )
                override = (sweep.parent.name, layer, run) == HEA_OVERRIDE
                if not reaches_4000 and not override:
                    notes.append(
                        f"Ignored unconfirmed incomplete run: layer {layer}, run {run} "
                        f"(metric={metric_epoch}, QQ={qq_epoch})"
                    )
                    continue
                if override:
                    notes.append(
                        "Included HEA layer 4, run 0 by explicit user confirmation; "
                        f"last metric={metric_epoch}, max QQ={qq_epoch}."
                    )

                epoch_path = metrics_folder / "loss_epochs.txt"
                loss_path = metrics_folder / "loss_wass.txt"
                epochs = losses = None
                if epoch_path.exists() and loss_path.exists():
                    epochs = np.atleast_1d(np.loadtxt(epoch_path, dtype=float))
                    losses = np.atleast_1d(np.loadtxt(loss_path, dtype=float))
                    length = min(len(epochs), len(losses))
                    epochs, losses = epochs[:length], losses[:length]
                    valid = np.isfinite(epochs) & np.isfinite(losses) & (losses > 0)
                    epochs, losses = epochs[valid], losses[valid]
                found[(layer, run)] = source.RunMetrics(
                    float(best[0]), int(round(best[1])), epochs, losses
                )
            except Exception as exc:
                notes.append(
                    f"Could not read metrics for layer {layer}, run {run}: {exc}"
                )
    return found, notes


def load_reconstruction_windows(
    sweep: Path,
) -> tuple[dict[tuple[int, int], np.ndarray], list[str]]:
    cache_folder = sweep / CACHE_FOLDER_NAME
    found: dict[tuple[int, int], np.ndarray] = {}
    notes: list[str] = []
    if not cache_folder.exists():
        return found, [f"Missing reconstruction cache folder: {cache_folder}"]
    for path in sorted(cache_folder.glob("generated_windows_layer_*_run_*.npz")):
        try:
            with np.load(path, allow_pickle=False) as npz:
                layer = int(np.asarray(npz["layer"]).reshape(-1)[0])
                run = int(np.asarray(npz["run_id"]).reshape(-1)[0])
                dataset = str(np.asarray(npz["dataset_file"]).reshape(-1)[0])
                if Path(dataset).name != source.DATASET:
                    notes.append(f"Ignored cache for {dataset}: {path}")
                    continue
                windows = source.as_windows(npz["generated_windows"], path)
            found[(layer, run)] = windows
        except Exception as exc:
            notes.append(f"Could not read {path}: {exc}")
    return found, notes


def completion_rows(
    model_name: str,
    sweep: Path,
    metrics: dict[tuple[int, int], source.RunMetrics],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for layer in source.LAYERS:
        roles = presentation.strict_roles(metrics, layer)
        role_by_run = {run: role for role, run in roles.items()} if roles else {}
        for run in source.RUNS:
            run_folder = (
                sweep / f"layers_{layer}_qubit_8_stride_1" / f"run_{run}"
            )
            metric_epoch, qq_epoch = _training_endpoint(run_folder)
            item = metrics.get((layer, run))
            rows.append(
                {
                    "model": model_name,
                    "layer": layer,
                    "run": run,
                    "role": role_by_run.get(run, "unranked"),
                    "best_wasserstein": item.best_loss if item else np.nan,
                    "best_epoch": item.best_epoch if item else np.nan,
                    "last_metric_epoch": metric_epoch,
                    "max_qq_epoch": qq_epoch,
                    "strict_epoch_4000": bool(
                        (metric_epoch is not None and metric_epoch >= 4000)
                        or (qq_epoch is not None and qq_epoch >= 4000)
                    ),
                    "user_confirmed_override": (
                        sweep.parent.name, layer, run
                    )
                    == HEA_OVERRIDE,
                }
            )
    return rows


def diagnostic_rows(
    model_name: str,
    sweep: Path,
    metrics: dict[tuple[int, int], source.RunMetrics],
    generated: dict[tuple[int, int], np.ndarray],
) -> list[dict[str, object]]:
    real = source.load_real_returns(sweep.parent)
    real_metrics = presentation.compute_window_metrics(
        source.chop_windows(real, stride=1)
    )
    real_centers = {
        name: np.nanmedian(values, axis=0)
        for name, values in real_metrics.items()
    }
    real_quantiles = np.quantile(real, source.QUANTILES)
    rows: list[dict[str, object]] = []
    for layer in source.LAYERS:
        roles = presentation.strict_roles(metrics, layer)
        if roles is None:
            continue
        run = roles["middle"]
        key = (layer, run)
        if key not in generated:
            continue
        windows = generated[key]
        generated_flat = windows.reshape(-1)
        window_metrics = presentation.compute_window_metrics(windows)
        generated_centers = {
            name: np.nanmedian(values, axis=0)
            for name, values in window_metrics.items()
        }
        row: dict[str, object] = {
            "model": model_name,
            "layer": layer,
            "middle_run": run,
            "training_seed": {0: 2, 1: 42, 2: 100}[run],
            "selected_run_best_wasserstein": metrics[key].best_loss,
            "selected_run_best_epoch": metrics[key].best_epoch,
            "evaluation_marginal_wasserstein": wasserstein_distance(
                real, generated_flat
            ),
            "qq_rmse": float(
                np.sqrt(
                    np.mean(
                        (
                            np.quantile(generated_flat, source.QUANTILES)
                            - real_quantiles
                        )
                        ** 2
                    )
                )
            ),
            "n_generated_windows": len(windows),
        }
        for metric_name in source.METRIC_LABELS:
            delta = generated_centers[metric_name] - real_centers[metric_name]
            row[f"{metric_name}_discrepancy_mae_lags_1_4"] = float(
                np.nanmean(np.abs(delta))
            )
            for lag_index, lag in enumerate(presentation.LAGS):
                row[f"real_{metric_name}_lag_{lag}"] = real_centers[metric_name][
                    lag_index
                ]
                row[f"generated_{metric_name}_lag_{lag}"] = generated_centers[
                    metric_name
                ][lag_index]
                row[f"delta_{metric_name}_lag_{lag}"] = delta[lag_index]
        rows.append(row)
    return rows


def write_analysis_summary(
    diagnostics: pd.DataFrame,
    completion: pd.DataFrame,
    output: Path,
) -> None:
    lines = [
        "# Complete four-model reconstruction: updated analysis",
        "",
        "All 48 model/layer/run combinations are included in run ranking and "
        "training-history analysis. HEA layer 4/run 0 is the single user-confirmed "
        "completion override (last recorded epoch 3900).",
        "",
        "## Median-run selections",
        "",
    ]
    for row in diagnostics.itertuples(index=False):
        lines.append(
            f"- {row.model}, layer {row.layer}: run {row.middle_run}; "
            f"saved best Wasserstein {row.selected_run_best_wasserstein:.6g}; "
            f"fresh evaluation Wasserstein {row.evaluation_marginal_wasserstein:.6g}."
        )
    lines.extend(["", "## Depth pattern", ""])
    middle = completion[completion["role"] == "middle"].copy()
    for model, group in middle.groupby("model", sort=False):
        group = group.sort_values("layer")
        best = group.loc[group["best_wasserstein"].idxmin()]
        values = ", ".join(
            f"L{int(row.layer)}={row.best_wasserstein:.6g}"
            for row in group.itertuples(index=False)
        )
        monotone = bool(np.all(np.diff(group["best_wasserstein"].to_numpy()) <= 0))
        lines.append(
            f"- {model}: {values}. Lowest median-run value at layer "
            f"{int(best['layer'])}; monotonic improvement with depth: "
            f"{'yes' if monotone else 'no'}."
        )
    lines.extend(["", "## Diagnostic comparison", ""])
    diagnostic_labels = {
        "evaluation_marginal_wasserstein": "fresh marginal Wasserstein distance",
        "abs_acf_discrepancy_mae_lags_1_4": "absolute-ACF discrepancy",
        "acf_discrepancy_mae_lags_1_4": "return-ACF discrepancy",
        "leverage_discrepancy_mae_lags_1_4": "leverage discrepancy",
    }
    for column, label in diagnostic_labels.items():
        best = diagnostics.loc[diagnostics[column].idxmin()]
        lines.append(
            f"- Lowest {label}: {best['model']} layer {int(best['layer'])} "
            f"({best[column]:.6g})."
        )

    zz_l1 = diagnostics[
        (diagnostics["model"] == "ZZ") & (diagnostics["layer"] == 1)
    ]
    evolution_l1 = diagnostics[
        (diagnostics["model"] == "ZZ time evolution")
        & (diagnostics["layer"] == 1)
    ]
    if len(zz_l1) == 1 and len(evolution_l1) == 1:
        compared = list(diagnostic_labels)
        identical = np.allclose(
            zz_l1.iloc[0][compared].to_numpy(dtype=float),
            evolution_l1.iloc[0][compared].to_numpy(dtype=float),
            rtol=0,
            atol=1e-15,
        )
        if identical:
            lines.append(
                "- ZZ and ZZ time evolution are numerically identical at layer 1 "
                "in all reconstructed diagnostics, as required by their shared "
                "single-layer circuit definition."
            )

    crx = diagnostics[diagnostics["model"] == "ZZ-CRX"].sort_values("layer")
    if len(crx):
        real_lag_1 = float(crx.iloc[0]["real_leverage_lag_1"])
        generated_values = ", ".join(
            f"L{int(row.layer)}={row.generated_leverage_lag_1:.4f}"
            for row in crx.itertuples(index=False)
        )
        lines.append(
            f"- ZZ-CRX lag-1 leverage estimates are {generated_values}, versus "
            f"Bitcoin={real_lag_1:.4f}. The intended negative direction is visible "
            "for layers 1–3 but is not uniform at layer 4 and does not establish "
            "overall superiority across all lags."
        )
    lines.extend(
        [
            "",
            "The saved best Wasserstein metric is the critic-independent marginal "
            "one-dimensional Wasserstein distance. Stylized-fact discrepancies are "
            "separate: each is the mean absolute difference across lags 1–4 between "
            "the real and generated median window-level curves.",
            "",
            "Confidence bands in the line figures are 95% max-t simultaneous "
            "bootstrap intervals across lags 1–4. Real data use a moving-block "
            "bootstrap before rebuilding stride-1 16-step windows; generated data "
            "resample complete 16-step windows.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    source.configure_plotting()
    source.load_run_metrics = load_run_metrics_complete
    source.load_generated_windows = load_reconstruction_windows
    source.load_real_returns = load_real_returns
    presentation.OUTPUT_FOLDER = OUTPUT_FOLDER
    presentation.REAL_INTERVAL_CACHE.clear()
    COMPARISON_FOLDER.mkdir(parents=True, exist_ok=True)

    manifests = []
    completion_data: list[dict[str, object]] = []
    diagnostic_data: list[dict[str, object]] = []
    for model_name, sweep in source.SWEEPS.items():
        manifest = presentation.analyze_model(
            model_name,
            sweep,
            wasserstein_filename="07_wasserstein_loss_vs_epoch_with_marker.png",
            wasserstein_plotter=plot_wasserstein_with_marker,
        )
        manifests.append(manifest)
        metrics, metric_notes = load_run_metrics_complete(sweep)
        generated, generated_notes = load_reconstruction_windows(sweep)
        completion_data.extend(completion_rows(model_name, sweep, metrics))
        diagnostic_data.extend(
            diagnostic_rows(model_name, sweep, metrics, generated)
        )

    completion = pd.DataFrame(completion_data)
    diagnostics = pd.DataFrame(diagnostic_data)
    completion.to_csv(
        COMPARISON_FOLDER / "all_run_completion_and_ranking.csv", index=False
    )
    diagnostics.to_csv(
        COMPARISON_FOLDER / "middle_run_diagnostic_summary.csv", index=False
    )
    write_analysis_summary(
        diagnostics,
        completion,
        COMPARISON_FOLDER / "updated_analysis_summary.md",
    )
    (COMPARISON_FOLDER / "analysis_manifests.json").write_text(
        json.dumps(manifests, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifests, indent=2))


if __name__ == "__main__":
    main()
