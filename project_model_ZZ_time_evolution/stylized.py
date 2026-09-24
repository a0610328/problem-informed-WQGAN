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

def calc_unbiased_window_metrics(data_2d, max_lag=8):
    """
    Calculates unbiased ACF and Leverage for short windows by anchoring 
    the centering process to the GLOBAL mean, avoiding Small Sample Bias.
    """
    # 1. Calculate GLOBAL means from the flattened 2D data
    flat_data = data_2d.flatten()
    global_mean_raw = np.mean(flat_data)           # Should be ~0
    global_mean_abs = np.mean(np.abs(flat_data))   # Strictly > 0
    global_mean_sq = np.mean(flat_data**2)
    
    acf_abs_list, acf_nonabs_list, lev_list = [], [], []
    T = data_2d.shape[1]
    N_windows = data_2d.shape[0]

    for window in data_2d:
        # 2. Center the window using GLOBAL means, not local means!
        centered_raw = window - global_mean_raw
        centered_abs = np.abs(window) - global_mean_abs
        centered_sq = (window**2) - global_mean_sq
        
        # Calculate variances for the denominator
        var_raw = np.sum(centered_raw**2)
        var_abs = np.sum(centered_abs**2)
        var_sq = np.sum(centered_sq**2)
        
        window_acf_abs, window_acf_nonabs, window_lev = [], [], []
        

        
        
        for lag in range(1, max_lag):
            unbiased_multiplier = T / (T - lag)

            # Absolute ACF (Volatility Clustering)
            cov_abs = np.sum(centered_abs[:-lag] * centered_abs[lag:])
            window_acf_abs.append(cov_abs / var_abs * unbiased_multiplier)
            
            # Non-absolute ACF (Raw Returns)
            cov_raw = np.sum(centered_raw[:-lag] * centered_raw[lag:])
            window_acf_nonabs.append(cov_raw / var_raw * unbiased_multiplier )
            
            # Leverage Effect (Past raw returns * Future absolute returns)
            # x_t is centered_raw, y_{t+lag} is centered_abs
            cov_lev = np.sum(centered_raw[:-lag] * centered_sq[lag:])
            # For cross-correlation, the denominator is the product of standard deviations
            denom_lev = np.sqrt(var_raw * var_sq)
            window_lev.append(cov_lev / denom_lev * unbiased_multiplier)
            
        acf_abs_list.append(window_acf_abs)
        acf_nonabs_list.append(window_acf_nonabs)
        lev_list.append(window_lev)
        
    # 3. Calculate final means and standard errors across all windows
    
    mean_abs = np.mean(acf_abs_list, axis=0)
    err_abs = np.std(acf_abs_list, axis=0, ddof=1) / np.sqrt(N_windows)
    
    mean_nonabs = np.mean(acf_nonabs_list, axis=0)
    err_nonabs = np.std(acf_nonabs_list, axis=0, ddof=1) / np.sqrt(N_windows)
    
    mean_lev = np.mean(lev_list, axis=0)
    err_lev = np.std(lev_list, axis=0, ddof=1) / np.sqrt(N_windows)
    
    return mean_abs, err_abs, mean_nonabs, err_nonabs, mean_lev, err_lev


def QQ_plot(data_1, data_2, title, xlabel, ylabel, limit, show = True, path = './'):
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
    ax.set_ylabel('Quantiles Bitcoin')
    ax.set_aspect('equal', adjustable='box')
    
    if show:
        plt.show()

    return fig

def compare_gen_vs_benchmark_unbiased(data_gen_2d, data_real_2d, run_id, n_layers, sweep_folder, metrics_real = None, max_lag=8, ylim_ACF=[0, 0.4], ylim_ACF_nonabs=[-0.15, 0.15], ylim_lev=[-0.15, 0.15], plot_stylized = True, is_show = False):
    
    print("Calculating Unbiased (T - lag) metrics internally...")
    
    # 1. 准备打平的数据用于 PDF & QQ-Plot
    data_gen = data_gen_2d.flatten()
    data_real = data_real_2d.flatten()
    N_windows= data_real_2d.shape[0]
    T = data_real_2d.shape[1]
    

    
    # 2. 分别计算生成数据和真实数据的窗口指标
    if metrics_real:
        ACF_bench, _, ACF_nonabs_bench, _, lev_bench, _ = metrics_real
    else:
        ACF_bench, _, ACF_nonabs_bench, _, lev_bench, _ = calc_unbiased_window_metrics(data_real_2d, max_lag)
    ACF_gen, ACF_err, ACF_nonabs_gen, ACF_nonabs_err, lev_gen, lev_err = calc_unbiased_window_metrics(data_gen_2d, max_lag)
    

    # 3. 动态置信区间 (因为使用了 T-lag，置信区间变成了喇叭口)
    lags = np.arange(1, max_lag)
    valid_counts = T - lags  # 计算每个 lag 真实参与计算的对数
    CI_dynamic = 1.96 / np.sqrt(valid_counts)/ np.sqrt(N_windows) # 动态置信区间
    
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
    plt.tight_layout()
    filename = f'PDF_Compare_run_{run_id}_layer_{n_layers}'
    plt.savefig(os.path.join(sweep_folder, filename))
    plt.savefig(os.path.join(sweep_folder, f"layers_{n_layers}_qubit_8_stride_1", f"run_{run_id}",filename))
    plt.tight_layout()
    if is_show:
        plt.show()

    # Plot 2: QQ-Plot 对比
    fig = QQ_plot(data_real, data_gen, 
        'Q-Q Plot (Generated vs Real)', 
        'Real Quantiles', 
        'Generated Quantiles', 
        limit = [-0.04,0.04], 
        show = True, 
        path='plot2_qq.pdf')
    filename = f'QQ_Plot_run_{run_id}_layer_{n_layers}'
    plt.savefig(os.path.join(sweep_folder, filename))
    plt.savefig(os.path.join(sweep_folder, f"layers_{n_layers}_qubit_8_stride_1", f"run_{run_id}",filename))
    plt.tight_layout()
    if is_show:
        plt.show()


    if plot_stylized:
        # Plot 3: 绝对收益率 ACF
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.bar(lags - width/2, ACF_bench, width=width, label='Bitcoin', alpha=0.7)
        ax3.bar(lags + width/2, ACF_gen, width=width, label='Generated', yerr=ACF_err, alpha=0.7)
        ax3.plot(lags, CI_dynamic, color='red', label=r'95% confidence interval')
        ax3.plot(lags, -CI_dynamic, color='red')
        ax3.set_title('ACF for absolute log returns')
        ax3.legend()
        ax3.set_ylabel('ACF')
        ax3.set_xlabel(r'Lag $\tau$')
        #ax3.set_ylim(ylim_ACF)
        filename = f'Absolute_ACF_run_{run_id}_layer_{n_layers}'
        plt.savefig(os.path.join(sweep_folder, filename))
        plt.savefig(os.path.join(sweep_folder, f"layers_{n_layers}_qubit_8_stride_1", f"run_{run_id}",filename))
        plt.tight_layout()
        if is_show:
            plt.show()
        
        # Plot 4: 对数收益率 ACF
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        ax4.bar(lags - width/2, ACF_nonabs_bench, width=width, label='Bitcoin', alpha=0.7)
        ax4.bar(lags + width/2, ACF_nonabs_gen, width=width, label='Generated', yerr=ACF_nonabs_err, alpha=0.7)
        ax4.plot(lags, CI_dynamic, color='red', label=r'95% confidence interval')
        ax4.plot(lags, -CI_dynamic, color='red')
        ax4.set_title('ACF for log returns')
        ax4.legend()
        ax4.set_ylabel('ACF')
        ax4.set_xlabel(r'Lag $\tau$')
        ax4.set_ylim(ylim_ACF_nonabs)
        filename = f'ACF_run_{run_id}_layer_{n_layers}'
        plt.savefig(os.path.join(sweep_folder, filename))
        plt.savefig(os.path.join(sweep_folder, f"layers_{n_layers}_qubit_8_stride_1", f"run_{run_id}",filename))
        plt.tight_layout()
        if is_show:
            plt.show()
        
        # Plot 5: 杠杆效应
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        ax5.bar(lags - width/2, lev_bench, width=width, label='Bitcoin', alpha=0.7)
        ax5.bar(lags + width/2, lev_gen, width=width, label='Generated', yerr=lev_err, alpha=0.7)
        ax5.plot(lags, CI_dynamic, color='red', label=r'95% confidence interval')
        ax5.plot(lags, -CI_dynamic, color='red')
        ax5.set_title('Leverage effect')
        ax5.legend()
        ax5.set_ylabel('Cross-Correlation')
        ax5.set_xlabel(r'Lag $\tau$')
        ax5.set_ylim(ylim_lev)
        filename = f'Leverage_run_{run_id}_layer_{n_layers}'
        plt.savefig(os.path.join(sweep_folder, filename))
        plt.savefig(os.path.join(sweep_folder, f"layers_{n_layers}_qubit_8_stride_1", f"run_{run_id}",filename))
        plt.tight_layout()
        if is_show:
            plt.show()

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
                # For n_layers=2 and shape=(n_layers,), we just pass a 1D array of the two parameters
                new_theta = np.array([t1, t2], dtype=np.float64)
                
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
                
                # COMPUTE: Evaluate new point
                # For n_layers=2 and shape=(n_layers,), we just pass a 1D array of the two parameters
                new_theta = np.array([t1, t2], dtype=np.float64)

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


