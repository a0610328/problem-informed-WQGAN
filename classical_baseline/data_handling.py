"""Local copy of the QGAN Bitcoin preprocessing required by the baselines."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import lambertw


def chopchop(data, window_size, stride=1):
    return np.lib.stride_tricks.sliding_window_view(
        data, window_shape=window_size
    )[::stride]


def transform(data, delta=0.5, unity_transform=False):
    """Apply the same two-stage normalization and inverse Lambert-W map."""

    mu1 = np.mean(data)
    s1 = np.std(data)
    data_norm1 = (data - mu1) / s1

    data_lambert = np.real(
        np.sign(data_norm1)
        * (lambertw(delta * data_norm1**2) / delta) ** 0.5
    )

    mu2 = np.mean(data_lambert)
    s2 = np.std(data_lambert)
    data_norm2 = (data_lambert - mu2) / s2
    minimum = np.quantile(data_norm2, 0.0005)
    maximum = np.quantile(data_norm2, 0.9995)

    if unity_transform:
        data_norm2 = 2.0 * (data_norm2 - minimum) / (maximum - minimum) - 1.0
        data_norm2 = np.clip(data_norm2, -1.0, 1.0)

    params = [
        mu1,
        s1,
        mu2,
        s2,
        delta,
        minimum,
        maximum,
        unity_transform,
    ]
    return data_norm2, params


def inverse_transform(data, params):
    mu1, s1, mu2, s2, delta, minimum, maximum, unity_transform = params

    if unity_transform:
        data_norm2 = (data + 1.0) / 2.0 * (maximum - minimum) + minimum
    else:
        clipped_below = np.where(data >= minimum, data, minimum)
        data_norm2 = np.where(clipped_below <= maximum, clipped_below, maximum)

    data_lambert = data_norm2 * s2 + mu2
    data_norm1 = data_lambert * np.exp(delta * data_lambert**2 / 2)
    return data_norm1 * s1 + mu1


def load_BTC_lr():
    """Load the bundled 2020--2026 daily Bitcoin log returns."""

    data_file = Path(__file__).resolve().parents[1] / "Bitcoin_Data_2020_2026" / "btc_daily_2020_2026.csv"
    if not data_file.exists():
        raise FileNotFoundError(
            f"Required data file is missing: {data_file}. Keep the "
            "Bitcoin_Data_2020_2026 folder alongside classical_baseline."
        )
    data_btc = pd.read_csv(data_file)
    if "Log_return" not in data_btc.columns:
        raise ValueError(f"{data_file} does not contain a Log_return column.")
    return data_btc["Log_return"].to_numpy(dtype=np.float64)
