"""Minimal local evaluation functions matching the QGAN experiments."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance


def metrics(generated_ts, real_ts):
    """Critic-independent marginal one-dimensional Wasserstein distance."""

    return wasserstein_distance(
        np.asarray(generated_ts).reshape(-1),
        np.asarray(real_ts).reshape(-1),
    )


def _validate_window_array(data_2d, max_lag):
    windows = np.asarray(data_2d, dtype=float)
    if windows.ndim != 2:
        raise ValueError("data_2d must have shape (n_windows, window_length).")
    if not 1 <= max_lag < windows.shape[1]:
        raise ValueError("max_lag must be between 1 and window_length - 1.")
    if not np.isfinite(windows).all():
        raise ValueError("data_2d contains non-finite values.")
    return windows


def _pooled_correlation(left, right):
    left = np.asarray(left, dtype=float).reshape(-1)
    right = np.asarray(right, dtype=float).reshape(-1)
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return np.nan
    return np.corrcoef(left, right)[0, 1]


def calc_window_respecting_pooled_metrics(data_2d, max_lag=7):
    """Calculate lag statistics without forming pairs across window boundaries."""

    windows = _validate_window_array(data_2d, max_lag)
    absolute_acf, raw_acf, leverage = [], [], []
    for lag in range(1, max_lag + 1):
        current = windows[:, :-lag]
        future = windows[:, lag:]
        raw_acf.append(_pooled_correlation(current, future))
        absolute_acf.append(_pooled_correlation(np.abs(current), np.abs(future)))
        leverage.append(_pooled_correlation(current, future**2))
    return (
        np.asarray(absolute_acf),
        np.asarray(raw_acf),
        np.asarray(leverage),
    )


def QQ_plot(data_1, data_2, title, xlabel, ylabel, limit, show=False):
    quantile_range = np.linspace(0, 1, 200)
    quantiles_1 = np.quantile(data_1, quantile_range)
    quantiles_2 = np.quantile(data_2, quantile_range)
    figure, axis = plt.subplots()
    axis.scatter(quantiles_1, quantiles_2)
    axis.plot(quantiles_1, quantiles_1, color="red")
    axis.set_xlim(limit)
    axis.set_ylim(limit)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_aspect("equal", adjustable="box")
    if show:
        plt.show()
    return figure

