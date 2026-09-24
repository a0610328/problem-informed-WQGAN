import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from statsmodels.tsa.stattools import acf, ccf

import pandas as pd
from scipy.special import lambertw
from scipy.stats import wasserstein_distance
import seaborn as sns


import tensorflow as tf
import data_handling as dh
import matplotlib.cbook as cbook
from arch import arch_model
from scipy.interpolate import griddata


# --- MATPLOTLIB 3.8+ COMPATIBILITY PATCH ---
# Prevents older mplot3d versions from crashing when calling the removed clean() method.
if not hasattr(cbook.Grouper, 'clean'):
    cbook.Grouper.clean = lambda self: None
# -------------------------------------------

# Apply the default theme
sns.set_theme()

def metrics(generated_ts, real_ts):
  EMD = wasserstein_distance(generated_ts.flatten(), real_ts.flatten())
  return EMD


GLOBAL_METRIC_COLUMNS = ("raw_acf", "absolute_acf", "leverage")


def _pearson_correlation(left, right):
    """Return a bounded sample correlation, or NaN for a degenerate pair."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = np.sqrt(np.dot(left_centered, left_centered) * np.dot(right_centered, right_centered))
    return np.nan if denominator == 0 else np.dot(left_centered, right_centered) / denominator


def compute_global_stylized_metrics(returns, max_lag=7):
    """Compute one full-series raw ACF, absolute ACF, and leverage curve.

    Unlike the short-window diagnostic, this uses the contiguous empirical
    return series and produces conventional bounded sample correlations.
    """
    returns = np.asarray(returns, dtype=float).reshape(-1)
    if len(returns) <= max_lag:
        raise ValueError("The return series must be longer than max_lag.")
    if not np.isfinite(returns).all():
        raise ValueError("The return series contains non-finite values.")

    rows = []
    absolute_returns = np.abs(returns)
    squared_returns = returns**2
    for lag in range(1, max_lag + 1):
        rows.append(
            {
                "lag": lag,
                "raw_acf": _pearson_correlation(returns[:-lag], returns[lag:]),
                "absolute_acf": _pearson_correlation(
                    absolute_returns[:-lag], absolute_returns[lag:]
                ),
                "leverage": _pearson_correlation(returns[:-lag], squared_returns[lag:]),
            }
        )
    return pd.DataFrame(rows)


def moving_block_bootstrap_stylized_metrics(
    returns, max_lag=7, block_length=30, n_bootstrap=2000, random_seed=42
):
    """Estimate percentile confidence intervals for full-series stylized metrics.

    The bootstrap preserves the ordering within each sampled block, unlike an
    IID bootstrap that would destroy volatility clustering and lag dependence.
    """
    returns = np.asarray(returns, dtype=float).reshape(-1)
    if not 1 <= block_length <= len(returns):
        raise ValueError("block_length must lie between 1 and the series length.")
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least 2.")

    point_estimate = compute_global_stylized_metrics(returns, max_lag=max_lag)
    bootstrap_values = np.empty((n_bootstrap, max_lag, len(GLOBAL_METRIC_COLUMNS)))
    generator = np.random.default_rng(random_seed)
    blocks_per_draw = int(np.ceil(len(returns) / block_length))
    last_start = len(returns) - block_length + 1

    # REVIEW: 30 daily observations is an initial choice that preserves the
    # seven evaluated lags and nearby volatility dependence. Check robustness
    # to alternative block lengths (for example 20, 40, and 60) before writing
    # a confidence-interval claim in the paper.
    for draw in range(n_bootstrap):
        # Sampling block starts with replacement retains within-block temporal
        # dependence while producing a return path of the original length.
        starts = generator.integers(0, last_start, size=blocks_per_draw)
        resampled_returns = np.concatenate(
            [returns[start : start + block_length] for start in starts]
        )[: len(returns)]
        bootstrap_metrics = compute_global_stylized_metrics(
            resampled_returns, max_lag=max_lag
        )
        bootstrap_values[draw] = bootstrap_metrics.loc[:, GLOBAL_METRIC_COLUMNS].to_numpy()

    interval_rows = {"lag": point_estimate["lag"].to_numpy()}
    for column_index, column in enumerate(GLOBAL_METRIC_COLUMNS):
        interval_rows[f"{column}_lower"] = np.quantile(
            bootstrap_values[:, :, column_index], 0.025, axis=0
        )
        interval_rows[f"{column}_upper"] = np.quantile(
            bootstrap_values[:, :, column_index], 0.975, axis=0
        )
    return point_estimate, pd.DataFrame(interval_rows)


def plot_global_stylized_metrics(point_estimate, intervals, data_period):
    """Plot global empirical stylized metrics with block-bootstrap intervals."""
    plot_definitions = (
        ("raw_acf", "Raw Returns Autocorrelation", "ACF"),
        ("absolute_acf", "Absolute Returns Autocorrelation", "ACF"),
        ("leverage", "Leverage Effect (Return vs Future Squared Return)", "Correlation"),
    )
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    lags = point_estimate["lag"].to_numpy()
    for axis, (column, title, ylabel) in zip(axes, plot_definitions):
        axis.axhline(0, color="black", linestyle="--", alpha=0.5)
        axis.plot(lags, point_estimate[column], marker="o", color="tab:blue", label="Estimate")
        axis.fill_between(
            lags,
            intervals[f"{column}_lower"].to_numpy(),
            intervals[f"{column}_upper"].to_numpy(),
            color="tab:blue",
            alpha=0.2,
            label="95% moving-block bootstrap CI",
        )
        axis.set_title(f"{title} - Empirical BTC Data ({data_period})")
        axis.set_ylabel(ylabel)
        axis.legend()
    axes[-1].set_xlabel("Lag (days)")
    fig.tight_layout()
    return fig

def _validate_window_array(data_2d, max_lag):
    """Validate a batch of temporal windows before pooling valid pairs."""
    windows = np.asarray(data_2d, dtype=float)
    if windows.ndim != 2:
        raise ValueError("data_2d must have shape (n_windows, window_length).")
    if not 1 <= max_lag < windows.shape[1]:
        raise ValueError("max_lag must be between 1 and window_length - 1.")
    return windows


def _pooled_correlation(left, right):
    """Pearson correlation after valid within-window pairs have been selected."""
    left = np.asarray(left, dtype=float).reshape(-1)
    right = np.asarray(right, dtype=float).reshape(-1)
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return np.nan
    return np.corrcoef(left, right)[0, 1]


def calc_window_respecting_pooled_metrics(data_2d, max_lag=7):
    """Compute QGAN-faithful stylized facts from valid pairs within each window.

    Flattening occurs only after selecting ``windows[:, :-lag]`` and
    ``windows[:, lag:]``. No temporal pair can cross a real or generated window
    boundary, and no ``T / (T - lag)`` finite-window multiplier is used.
    """
    windows = _validate_window_array(data_2d, max_lag)
    absolute_acf, raw_acf, leverage = [], [], []
    for lag in range(1, max_lag + 1):
        current = windows[:, :-lag]
        future = windows[:, lag:]
        raw_acf.append(_pooled_correlation(current, future))
        absolute_acf.append(_pooled_correlation(np.abs(current), np.abs(future)))
        leverage.append(_pooled_correlation(current, future ** 2))
    return np.asarray(absolute_acf), np.asarray(raw_acf), np.asarray(leverage)


def bootstrap_window_respecting_pooled_metrics(data_2d, max_lag=7, n_bootstrap=500, seed=42):
    """Bootstrap whole windows for uncertainty on the pooled-pair estimator.

    With overlapping real windows, this is conditional resampling of the supplied
    window set, not an independent time-series confidence interval.
    """
    windows = _validate_window_array(data_2d, max_lag)
    rng = np.random.default_rng(seed)
    draws = np.empty((n_bootstrap, 3, max_lag), dtype=float)
    for draw in range(n_bootstrap):
        sample = windows[rng.integers(0, len(windows), len(windows))]
        draws[draw] = calc_window_respecting_pooled_metrics(sample, max_lag)
    return np.nanstd(draws, axis=0, ddof=1)


def calc_unbiased_window_metrics(data_2d, max_lag=7):
    """Backward-compatible wrapper for the pooled within-window estimator."""
    mean_abs, mean_nonabs, mean_lev = calc_window_respecting_pooled_metrics(data_2d, max_lag)
    no_error = np.full(max_lag, np.nan)
    return mean_abs, no_error, mean_nonabs, no_error.copy(), mean_lev, no_error.copy()


def calc_window_metric_distributions(data_2d, max_lag=7):
    """Return distributions of individual window estimates for violin diagnostics.

    These are intentionally local Pearson estimates, rather than pooled scores.
    They describe heterogeneity across 16-day windows and should be labelled as
    such because their spread is not an independent-sample confidence interval.
    """
    windows = _validate_window_array(data_2d, max_lag)
    raw = np.full((len(windows), max_lag), np.nan)
    absolute = np.full_like(raw, np.nan)
    leverage = np.full_like(raw, np.nan)
    for index, window in enumerate(windows):
        for lag in range(1, max_lag + 1):
            current, future = window[:-lag], window[lag:]
            raw[index, lag - 1] = _pooled_correlation(current, future)
            absolute[index, lag - 1] = _pooled_correlation(np.abs(current), np.abs(future))
            leverage[index, lag - 1] = _pooled_correlation(current, future ** 2)
    return absolute, raw, leverage


def QQ_plot(data_1, data_2, title, xlabel, ylabel, limit, show=False):
    Q_range = np.linspace(0,1,200)
    Q_data_1 = np.quantile(data_1, Q_range)
    Q_data_2 = np.quantile(data_2, Q_range)
    fig, ax = plt.subplots()  # 创建明确的 Figure 和 Axes 对象
    ax.scatter(Q_data_1, Q_data_2)
    ax.plot(Q_data_1, Q_data_1, color='red')
    ax.set_xlim(limit)
    ax.set_ylim(limit)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect('equal', adjustable='box')
    
    if show:
        plt.show()

    return fig

def compare_gen_vs_benchmark_unbiased(data_gen_2d, data_real_2d, run_id, n_layers, sweep_folder, metrics_real=None, max_lag=7, ylim_ACF=[0, 0.4], ylim_ACF_nonabs=[-0.15, 0.15], ylim_lev=[-0.15, 0.15], plot_stylized=True, is_show=False, stride=1, n_qubits=None, n_bootstrap=500):
    """Compare matched real and generated windows without crossing boundaries.

    ``metrics_real`` is retained only for call compatibility and is deliberately
    ignored: both inputs are recomputed with the same pooled-pair estimator.
    """
    print("Calculating window-respecting pooled-pair metrics internally...")
    data_gen_2d = _validate_window_array(data_gen_2d, max_lag)
    data_real_2d = _validate_window_array(data_real_2d, max_lag)
    if data_gen_2d.shape != data_real_2d.shape:
        raise ValueError("Generated and real data must contain the same number of equally sized windows.")
    
    # 1. 准备打平的数据用于 PDF & QQ-Plot
    data_gen = data_gen_2d.flatten()
    data_real = data_real_2d.flatten()
    T = data_real_2d.shape[1]
    n_qubits = n_qubits if n_qubits is not None else T // 2
    run_folder = os.path.join(sweep_folder, f"layers_{n_layers}_qubit_{n_qubits}_stride_{stride}", f"run_{run_id}")
    

    
    # 2. Same estimator for real and generated data: valid pairs within windows only.
    ACF_bench, ACF_nonabs_bench, lev_bench = calc_window_respecting_pooled_metrics(data_real_2d, max_lag)
    ACF_gen, ACF_nonabs_gen, lev_gen = calc_window_respecting_pooled_metrics(data_gen_2d, max_lag)
    real_errors = bootstrap_window_respecting_pooled_metrics(data_real_2d, max_lag, n_bootstrap, seed=42)
    gen_errors = bootstrap_window_respecting_pooled_metrics(data_gen_2d, max_lag, n_bootstrap, seed=43)
    ACF_real_err, ACF_nonabs_real_err, lev_real_err = real_errors
    ACF_err, ACF_nonabs_err, lev_err = gen_errors
    

    # Bootstrap standard errors describe resampling uncertainty in the supplied windows.
    lags = np.arange(1, max_lag + 1)
    CI_dynamic = {"real": real_errors, "generated": gen_errors}
    
    width = 0.35

    print("Plotting perfectly aligned unbiased comparisons...")
    
    # ================= 绘图部分 =================
    
    # Plot 1: PDF 分布对比
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.kdeplot(data_real, label='Bitcoin (Real)', ax=ax1, color='blue', fill=True, alpha=0.3)
    sns.kdeplot(data_gen, label='Generated', ax=ax1, color='orange', fill=True, alpha=0.3)
    ax1.set_title('Probability Density Function (PDF) of Returns')
    ax1.set_xlabel('Log Returns')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.set_xlim([-0.2, 0.2]) 
    fig1.tight_layout()
    filename = f'PDF_Compare_run_{run_id}_layer_{n_layers}'
    fig1.savefig(os.path.join(sweep_folder, filename))
    fig1.savefig(os.path.join(run_folder, filename))
    if is_show:
        plt.show()
    plt.close(fig1)

    # Plot 2: QQ-Plot 对比
    fig = QQ_plot(data_real, data_gen, 
        'Bitcoin Log-Return Q-Q Comparison (Generated vs Real)', 
        'Real Bitcoin log-return quantiles', 
        'Generated Bitcoin log-return quantiles', 
        limit = [-0.04,0.04], 
        show = False, 
    )
    filename = f'QQ_Plot_run_{run_id}_layer_{n_layers}.png'
    fig.tight_layout()
    fig.savefig(os.path.join(sweep_folder, filename))
    fig.savefig(os.path.join(run_folder, filename))
    

    if is_show:
        plt.show()
    plt.close(fig)


    if plot_stylized:
        # Plot 3: 绝对收益率 ACF
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.bar(lags - width/2, ACF_bench, width=width, label='Bitcoin', yerr=ACF_real_err, alpha=0.7)
        ax3.bar(lags + width/2, ACF_gen, width=width, label='Generated', yerr=ACF_err, alpha=0.7)
        ax3.set_title('Window-Respecting Pooled-Pair ACF for Absolute Log Returns')
        ax3.legend()
        ax3.set_ylabel('Correlation')
        ax3.set_xlabel(r'Lag $\tau$')
        ax3.set_ylim(ylim_ACF)
        filename = f'Absolute_ACF_run_{run_id}_layer_{n_layers}'
        fig3.tight_layout()
        fig3.savefig(os.path.join(sweep_folder, filename))
        fig3.savefig(os.path.join(run_folder, filename))
        if is_show:
            plt.show()
        plt.close(fig3)
        
        # Plot 4: 对数收益率 ACF
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        ax4.bar(lags - width/2, ACF_nonabs_bench, width=width, label='Bitcoin', yerr=ACF_nonabs_real_err, alpha=0.7)
        ax4.bar(lags + width/2, ACF_nonabs_gen, width=width, label='Generated', yerr=ACF_nonabs_err, alpha=0.7)
        ax4.set_title('Window-Respecting Pooled-Pair ACF for Log Returns')
        ax4.legend()
        ax4.set_ylabel('Correlation')
        ax4.set_xlabel(r'Lag $\tau$')
        ax4.set_ylim(ylim_ACF_nonabs)
        filename = f'ACF_run_{run_id}_layer_{n_layers}'
        fig4.tight_layout()
        fig4.savefig(os.path.join(sweep_folder, filename))
        fig4.savefig(os.path.join(run_folder, filename))
        if is_show:
            plt.show()
        plt.close(fig4)
        
        # Plot 5: 杠杆效应
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        ax5.bar(lags - width/2, lev_bench, width=width, label='Bitcoin', yerr=lev_real_err, alpha=0.7)
        ax5.bar(lags + width/2, lev_gen, width=width, label='Generated', yerr=lev_err, alpha=0.7)
        ax5.set_title('Window-Respecting Pooled-Pair Leverage Effect')
        ax5.legend()
        ax5.set_ylabel('Cross-Correlation')
        ax5.set_xlabel(r'Lag $\tau$')
        ax5.set_ylim(ylim_lev)
        filename = f'Leverage_run_{run_id}_layer_{n_layers}'
        fig5.tight_layout()
        fig5.savefig(os.path.join(sweep_folder, filename))
        fig5.savefig(os.path.join(run_folder, filename))
        if is_show:
            plt.show()
        plt.close(fig5)

    return CI_dynamic, ACF_bench, ACF_gen, ACF_nonabs_bench, ACF_nonabs_gen, lev_bench, lev_gen



def plot_true_wasserstein_landscape(gan_instance, batch_size=1000, resolution=20, save_folder = './'):
    """
    Evaluates and plots the True Wasserstein loss energy landscape for the quantum generator.
    Reuses previously computed points from CSV and only calculates new grid coordinates.
    
    Args:
        gan_instance: An initialized instance of your quantum_GAN class.
        batch_size: Number of noise samples for EMD evaluation.
        resolution: Target grid resolution (e.g., 50 for a 50x50 mesh).
        filename: CSV filename for reading/saving landscape data.
    """
    
    # 1. Define target 50x50 grid coordinates [0, 2pi]
    theta_1_vals = np.linspace(0, 2 * np.pi, resolution)
    theta_2_vals = np.linspace(0, 2 * np.pi, resolution)
    
    T1, T2 = np.meshgrid(theta_1_vals, theta_2_vals)
    Loss_Grid = np.zeros_like(T1)
    
    # Dictionary to hold evaluated coordinates -> key: (round(t1, 5), round(t2, 5)), value: EMD
    cache_dict = {}

    filename="true_emd_landscape_data.csv"
    # 2. Load existing points if CSV exists
    if os.path.exists(os.path.join(save_folder,filename)):
        existing_df = pd.read_csv(os.path.join(save_folder,filename))
        print(f"Loaded {len(existing_df)} existing points from '{filename}'.")
        for _, row in existing_df.iterrows():
            key = (round(row['Theta_1'], 5), round(row['Theta_2'], 5))
            cache_dict[key] = row['True_EMD']
    else:
        print(f"No existing cache found. Creating new dataset...")

    # 3. Setup Quantum Generator for evaluating missing points
    pqc_layer = gan_instance.generator.get_layer('re-uploading_PQC')
    original_weights = pqc_layer.get_weights()
    n_layers = gan_instance.n_layers
    
    static_noise = gan_instance.get_noise(batch_size) 
    real_data = gan_instance.data_real_2d 
    
    new_evaluations = 0
    total_points = resolution * resolution
    
    print(f"Building/Updating {resolution}x{resolution} grid ({total_points} total points)...")
    
    # 4. Iterate through the target mesh grid
    for i in range(resolution):
        for j in range(resolution):
            t1 = T1[i, j]
            t2 = T2[i, j]
            key = (round(t1, 5), round(t2, 5))
            
            # REUSE: Check if point was previously computed
            if key in cache_dict:
                Loss_Grid[i, j] = cache_dict[key]
            else:
                # COMPUTE: Evaluate new point
                new_theta = np.array([[t1] * n_layers, 
                                      [t2] * n_layers], dtype=np.float64)
                
                pqc_layer.set_weights([new_theta])
                
                generated_data_raw = gan_instance.generator(static_noise, training=False).numpy()
                generated_data_scaled = generated_data_raw * gan_instance.scale_factor
                generated_data_transformed = dh.inverse_transform(generated_data_scaled, gan_instance.transform_params)
                
                true_emd = metrics(generated_data_transformed, real_data)
                
                # Store in cache and grid
                cache_dict[key] = true_emd
                Loss_Grid[i, j] = true_emd
                new_evaluations += 1

    # Restore generator's weights
    pqc_layer.set_weights(original_weights)
    
    print(f"Evaluation complete! Reused points: {total_points - new_evaluations}, Newly computed: {new_evaluations}.")
    
    # 5. Save all combined points back to CSV
    export_data = [{"Theta_1": k[0], "Theta_2": k[1], "True_EMD": v} for k, v in cache_dict.items()]
    df_landscape = pd.DataFrame(export_data)
    df_landscape.to_csv(os.path.join(save_folder,filename), index=False)
    print(f"Saved total of {len(df_landscape)} parameter-loss pairs to '{filename}'")

    # 6. Render Interactive Plotly Visualization
    pi_ticks = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
    pi_ticktexts = ['0', 'π/2', 'π', '3π/2', '2π']

    fig = go.Figure(data=[go.Surface(
        z=Loss_Grid, 
        x=T1, 
        y=T2, 
        colorscale='Magma',
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
        )
    )])

    fig.update_layout(
        title=f'Interactive True Wasserstein Landscape ({resolution}x{resolution})',
        scene=dict(
            xaxis=dict(title='Parameter 1 (Theta 1)', tickvals=pi_ticks, ticktext=pi_ticktexts),
            yaxis=dict(title='Parameter 2 (Theta 2)', tickvals=pi_ticks, ticktext=pi_ticktexts),
            zaxis=dict(title='True EMD')
        ),
        width=900, 
        height=800
    )

    fig.write_html("interactive_true_emd_landscape.html")
    fig.show()
    
    return df_landscape


import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from arch import arch_model
import data_handling as dh

def plot_garch_landscape(gan_instance, batch_size=1000, resolution=50, folder='./'):
    """
    Evaluates and plots the Euclidean distance between real and generated 
    GARCH(1,1) parameters (alpha and beta) directly sampling over:
      - Theta 1 in [-pi/2, 3pi/2]
      - Theta 2 in [-pi, pi]
      
    All file read and write operations (CSV data and interactive HTML plots)
    are saved to or loaded from `folder`.
    """
    
    # --- FIXED FILENAMES AND PATHS ---
    filename = "garch_landscape_direct.csv"
    html_filename = "interactive_garch_landscape_direct.html"
    
    csv_path = os.path.join(folder, filename)
    html_path = os.path.join(folder, html_filename)
    
    # Ensure destination folder exists
    os.makedirs(folder, exist_ok=True)


    # 1. Baseline: Fit GARCH(1,1) to Real Data
    real_data = gan_instance.data_real_2d 
    flat_real = real_data.flatten()
    
    # Dynamically scale real data to have a variance of exactly 100
    std_real = np.std(flat_real)
    scale_real = 10.0 / std_real if std_real > 0 else 1.0
    
    am_real = arch_model(flat_real * scale_real, vol='GARCH', p=1, q=1, dist='Normal')
    res_real = am_real.fit(disp='off')
    
    alpha_real = res_real.params['alpha[1]']
    beta_real = res_real.params['beta[1]']
    print(f"Real Bitcoin GARCH Params -> Alpha: {alpha_real:.4f}, Beta: {beta_real:.4f}")

    # 2. Define Direct Parameter Space
    theta_1_vals = np.linspace(-np.pi / 2.0, 1.5 * np.pi, resolution)
    theta_2_vals = np.linspace(-np.pi, np.pi, resolution)
    
    T1, T2 = np.meshgrid(theta_1_vals, theta_2_vals)
    Loss_Grid = np.full(T1.shape, np.nan)

    # 3. Cache Check / Evaluation Loop
    if os.path.exists(csv_path):
        print(f"Loading direct GARCH landscape data from '{csv_path}'...")
        df = pd.read_csv(csv_path)
        pivot_table = df.pivot(index='Theta_2', columns='Theta_1', values='GARCH_Distance')
        T1, T2 = np.meshgrid(pivot_table.columns.values, pivot_table.index.values)
        Loss_Grid = pivot_table.values
    else:
        print(f"Evaluating GARCH(1,1) distance directly across {resolution}x{resolution} grid...")
        
        pqc_layer = gan_instance.generator.get_layer('re-uploading_PQC')
        original_weights = pqc_layer.get_weights()
        n_layers = gan_instance.n_layers
        static_noise = gan_instance.get_noise(batch_size) 
        landscape_data = []

        for i in range(resolution):
            for j in range(resolution):
                t1 = T1[i, j]
                t2 = T2[i, j]
                
                # Direct injection into quantum generator
                new_theta = np.array([[t1] * n_layers, [t2] * n_layers], dtype=np.float64)
                pqc_layer.set_weights([new_theta])
                
                # Forward pass & inverse transformation
                generated_data_raw = gan_instance.generator(static_noise, training=False).numpy()
                generated_data_scaled = generated_data_raw * gan_instance.scale_factor
                generated_data_transformed = dh.inverse_transform(generated_data_scaled, gan_instance.transform_params)
                
                flat_gen = generated_data_transformed.flatten()

                # Fit GARCH to generated data (with dynamic scaling)
                try:
                    std_gen = np.std(flat_gen)
                    
                    # If the quantum circuit outputs a flat line, skip fitting and penalize
                    if std_gen < 1e-8:
                        distance = 2.0
                    else:
                        # Dynamically scale generated data to have a variance of exactly 100
                        scale_gen = 10.0 / std_gen
                        
                        am_gen = arch_model(flat_gen * scale_gen, vol='GARCH', p=1, q=1, dist='Normal')
                        res_gen = am_gen.fit(disp='off')
                        
                        alpha_gen = res_gen.params['alpha[1]']
                        beta_gen = res_gen.params['beta[1]']
                        
                        # Calculate Euclidean Distance
                        distance = np.sqrt((alpha_real - alpha_gen)**2 + (beta_real - beta_gen)**2)
                        
                except Exception:
                    distance = 2.0  # Penalty value for degenerate runs
                
                Loss_Grid[i, j] = distance
                landscape_data.append({"Theta_1": t1, "Theta_2": t2, "GARCH_Distance": distance})

        # Restore original generator weights
        pqc_layer.set_weights(original_weights)
        
        # Save exact direct coordinates
        df = pd.DataFrame(landscape_data)
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(df)} points to '{csv_path}'.")

    # 4. Axis Ticks & Labels
    t1_ticks = [-np.pi/2, 0, np.pi/2, np.pi, 1.5 * np.pi]
    t1_ticktexts = ['-π/2', '0', 'π/2', 'π', '3π/2']

    t2_ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
    t2_ticktexts = ['-π', '-π/2', '0', 'π/2', 'π']

    z_min, z_max = np.nanmin(Loss_Grid), np.nanmax(Loss_Grid)
    contour_step = (z_max - z_min) / 20.0

    # 5. Render Surface directly
    fig = go.Figure(data=[go.Surface(
        z=Loss_Grid, 
        x=T1, 
        y=T2, 
        colorscale='Viridis',
        lighting=dict(ambient=0.6, diffuse=0.8, fresnel=0.2, specular=0.5, roughness=0.3),
        contours=dict(
            z=dict(
                show=True, 
                usecolormap=True, 
                highlightcolor="red", 
                project_z=True, 
                start=z_min, 
                end=z_max, 
                size=contour_step
            )
        )
    )])

    fig.update_layout(
        title='Direct GARCH(1,1) Parameter Distance Landscape<br><sup>(Target: Euclidean distance between Alpha and Beta)</sup>',
        scene=dict(
            xaxis=dict(title='Parameter 1 (Theta 1)', tickvals=t1_ticks, ticktext=t1_ticktexts),
            yaxis=dict(title='Parameter 2 (Theta 2)', tickvals=t2_ticks, ticktext=t2_ticktexts),
            zaxis=dict(title="GARCH Distance")
        ),
        width=950, 
        height=850, 
        margin=dict(l=65, r=50, b=65, t=90)
    )

    fig.write_html(html_path)
    fig.show()

    return Loss_Grid


import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from stylized import calc_unbiased_window_metrics

def generate_ideal_stylized_facts(real_data_2d, max_lag=7):
    """
    Calculates d, gamma, and delta using standard Python ARCH models,
    simulates the ideal path, and computes the theoretical stylized facts.
    """
    flat_real = real_data_2d.flatten()
    
    # ---------------------------------------------------------
    # 1. Parameter Extraction: Gamma and Delta (APARCH)
    # ---------------------------------------------------------
    print("Fitting APARCH model for Asymmetry (Gamma) and Power (Delta)...")
    am_aparch = arch_model(flat_real, vol='APARCH', p=1, o=1, q=1, dist='Normal')
    res_aparch = am_aparch.fit(disp='off')
    
    gamma = res_aparch.params.get('gamma[1]', 0.0)
    delta = res_aparch.params.get('power', 2.0)
    
    print(f"Estimated Gamma (Asymmetry): {gamma:.4f}")
    print(f"Estimated Delta (Power):     {delta:.4f}")

    # ---------------------------------------------------------
    # 2. Parameter Extraction: d (FIGARCH)
    # ---------------------------------------------------------
    print("Fitting FIGARCH model for Fractional Integration (d)...")
    am_figarch = arch_model(flat_real, vol='FIGARCH', p=1, q=1, dist='Normal')
    res_figarch = am_figarch.fit(disp='off')
    
    d = res_figarch.params.get('d', 0.0)
    print(f"Estimated d (Long Memory):   {d:.4f}")

    # ---------------------------------------------------------
    # 3. Simulate the Ideal Path
    # ---------------------------------------------------------
    print("\nSimulating ideal, noise-free time series path...")
    # Generate a massive synthetic path to ensure convergence to theoretical bounds
    n_simulations = max(len(flat_real) * 10, 20000)
    
    # Simulating using APARCH to capture the critical leverage asymmetry
    simulated_data = am_aparch.simulate(res_aparch.params, nobs=n_simulations)
    
    # Reshape the 1D simulation into the 2D window format expected by stylized.py
    window_size = real_data_2d.shape[1]
    valid_length = (n_simulations // window_size) * window_size
    sim_2d = simulated_data['data'].values[:valid_length].reshape(-1, window_size)

    # ---------------------------------------------------------
    # 4. Calculate Ideal Properties using stylized.py
    # ---------------------------------------------------------
    print("Calculating unbiased window metrics on theoretical data...")
    metrics_ideal = calc_unbiased_window_metrics(sim_2d, max_lag=max_lag)
    
    mean_abs_ideal = metrics_ideal[0]
    mean_nonabs_ideal = metrics_ideal[2]
    mean_lev_ideal = metrics_ideal[4]

    # Calculate real data metrics for side-by-side plotting
    metrics_real = calc_unbiased_window_metrics(real_data_2d, max_lag=max_lag)
    mean_abs_real = metrics_real[0]
    mean_nonabs_real = metrics_real[2]
    mean_lev_real = metrics_real[4]

    # ---------------------------------------------------------
    # 5. Plotting Comparison
    # ---------------------------------------------------------
    lags = np.arange(1, max_lag + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    width = 0.35
    
    # Absolute ACF
    axes[0].bar(lags - width/2, mean_abs_real, width=width, label='Real Bitcoin Data', alpha=0.7)
    axes[0].bar(lags + width/2, mean_abs_ideal, width=width, label='Ideal APARCH Model', alpha=0.7)
    axes[0].set_title('Absolute ACF (Volatility Clustering)')
    axes[0].legend()
    
    # Raw ACF
    axes[1].bar(lags - width/2, mean_nonabs_real, width=width, label='Real', alpha=0.7)
    axes[1].bar(lags + width/2, mean_nonabs_ideal, width=width, label='Ideal APARCH', alpha=0.7)
    axes[1].set_title('ACF (Raw Returns)')
    
    # Leverage Effect
    axes[2].bar(lags - width/2, mean_lev_real, width=width, label='Real', alpha=0.7)
    axes[2].bar(lags + width/2, mean_lev_ideal, width=width, label='Ideal APARCH', alpha=0.7)
    axes[2].set_title('Leverage Effect (Cross-Correlation)')

    plt.tight_layout()
    plt.show()

    return d, gamma, delta, mean_abs_ideal, mean_nonabs_ideal, mean_lev_ideal
