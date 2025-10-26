#!/usr/bin/env python
"""
Phase Space Analysis for Quantum Tunneling
Re(ψ) vs Im(ψ) - Clean scatter visualization with robust statistics
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.spatial import ConvexHull
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def stratified_sample_2d(x, y, n_samples, n_strata=10):
    """2D stratified sampling preserving joint distribution."""
    if len(x) <= n_samples:
        return x, y
    
    x_edges = np.percentile(x, np.linspace(0, 100, n_strata + 1))
    y_edges = np.percentile(y, np.linspace(0, 100, n_strata + 1))
    
    sampled_x = []
    sampled_y = []
    samples_per_stratum = n_samples // (n_strata * n_strata)
    
    for i in range(n_strata):
        for j in range(n_strata):
            x_mask = (x >= x_edges[i]) & (x < x_edges[i + 1])
            y_mask = (y >= y_edges[j]) & (y < y_edges[j + 1])
            stratum_mask = x_mask & y_mask
            
            x_stratum = x[stratum_mask]
            y_stratum = y[stratum_mask]
            
            if len(x_stratum) > 0:
                n_sample = min(samples_per_stratum, len(x_stratum))
                indices = np.random.choice(len(x_stratum), n_sample, replace=False)
                sampled_x.extend(x_stratum[indices])
                sampled_y.extend(y_stratum[indices])
    
    return np.array(sampled_x), np.array(sampled_y)


def phase_space_entropy(psi_real, psi_imag, bins=50):
    """Calculate 2D phase space entropy."""
    hist_2d, _, _ = np.histogram2d(psi_real, psi_imag, bins=bins)
    hist_2d = hist_2d[hist_2d > 0]
    prob_2d = hist_2d / np.sum(hist_2d)
    entropy = -np.sum(prob_2d * np.log2(prob_2d))
    return entropy


def circular_variance(phases):
    """Circular variance for phase angle distribution."""
    mean_direction = np.mean(np.exp(1j * phases))
    R = np.abs(mean_direction)
    return 1 - R, R


def phase_coherence_index(psi_real, psi_imag):
    """Phase coherence index: PCI = |⟨ψ⟩| / ⟨|ψ|⟩"""
    psi_complex = psi_real + 1j * psi_imag
    mean_psi = np.mean(psi_complex)
    mean_abs_psi = np.mean(np.abs(psi_complex))
    
    if mean_abs_psi > 0:
        pci = np.abs(mean_psi) / mean_abs_psi
    else:
        pci = 0
    
    return pci


def anisotropy_ratio(psi_real, psi_imag):
    """Anisotropy ratio from covariance matrix eigenvalues."""
    cov_matrix = np.cov(psi_real, psi_imag)
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    
    lambda_max = np.max(eigenvalues)
    lambda_min = np.min(eigenvalues)
    
    if lambda_max + lambda_min > 0:
        ar = (lambda_max - lambda_min) / (lambda_max + lambda_min)
    else:
        ar = 0
    
    return ar, eigenvalues


def ellipticity_measure(psi_real, psi_imag):
    """Ellipticity: e = sqrt(1 - b²/a²)"""
    cov_matrix = np.cov(psi_real, psi_imag)
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    
    a = np.sqrt(np.max(eigenvalues))
    b = np.sqrt(np.min(eigenvalues))
    
    if a > 0:
        ellipticity = np.sqrt(1 - (b**2 / a**2))
    else:
        ellipticity = 0
    
    return ellipticity, a, b


def phase_space_area(psi_real, psi_imag, quantile=0.95):
    """Effective phase space area via convex hull."""
    r = np.sqrt(psi_real**2 + psi_imag**2)
    r_threshold = np.quantile(r, quantile)
    
    mask = r <= r_threshold
    points = np.column_stack([psi_real[mask], psi_imag[mask]])
    
    if len(points) > 3:
        try:
            hull = ConvexHull(points)
            area = hull.volume
        except:
            area = 0
    else:
        area = 0
    
    return area


def radial_distribution_stats(psi_real, psi_imag):
    """Statistics of radial distribution r = |ψ|."""
    r = np.sqrt(psi_real**2 + psi_imag**2)
    
    stats_dict = {
        'mean': np.mean(r),
        'std': np.std(r),
        'median': np.median(r),
        'q25': np.percentile(r, 25),
        'q75': np.percentile(r, 75),
        'iqr': np.percentile(r, 75) - np.percentile(r, 25),
        'skewness': stats.skew(r),
        'kurtosis': stats.kurtosis(r)
    }
    
    return r, stats_dict


def angular_distribution_stats(psi_real, psi_imag):
    """Statistics of angular distribution θ = arg(ψ)."""
    theta = np.arctan2(psi_imag, psi_real)
    
    circular_mean = np.angle(np.mean(np.exp(1j * theta)))
    circ_var, concentration = circular_variance(theta)
    
    stats_dict = {
        'circular_mean': circular_mean,
        'circular_variance': circ_var,
        'concentration': concentration,
        'std_dev': stats.circstd(theta),
        'entropy': phase_space_entropy(np.cos(theta), np.sin(theta), bins=36)
    }
    
    return theta, stats_dict


def mutual_information_phase_space(psi_real, psi_imag, bins=50):
    """Mutual information I(Re;Im)."""
    hist_re, _ = np.histogram(psi_real, bins=bins, density=True)
    hist_im, _ = np.histogram(psi_imag, bins=bins, density=True)
    
    prob_re = hist_re / (np.sum(hist_re) + 1e-12)
    prob_im = hist_im / (np.sum(hist_im) + 1e-12)
    
    prob_re = prob_re[prob_re > 0]
    prob_im = prob_im[prob_im > 0]
    
    H_re = -np.sum(prob_re * np.log2(prob_re))
    H_im = -np.sum(prob_im * np.log2(prob_im))
    H_joint = phase_space_entropy(psi_real, psi_imag, bins=bins)
    
    mi = H_re + H_im - H_joint
    
    return mi, H_re, H_im, H_joint


def phase_space_moment(psi_real, psi_imag, p, q):
    """Phase space moment M_pq = ⟨Re^p * Im^q⟩."""
    moment = np.mean((psi_real ** p) * (psi_imag ** q))
    return moment


def main():
    np.random.seed(42)
    
    print("\n" + "=" * 80)
    print("PHASE SPACE ANALYSIS: Re(ψ) vs Im(ψ)")
    print("=" * 80)
    
    # Load data
    print("\n[1/5] Loading datasets...")
    rec_barrier = xr.open_dataset("../raw_data/case1_rectangular_barrier.nc")
    gauss_barrier = xr.open_dataset("../raw_data/case2_gaussian_barrier.nc")
    print("✓ Datasets loaded")
    
    # Create output directories
    Path("../figs").mkdir(parents=True, exist_ok=True)
    Path("../stats").mkdir(parents=True, exist_ok=True)
    
    # Extract wavefunction components
    print("\n[2/5] Extracting wavefunction components...")
    psi_real_rect = rec_barrier['psi_real'].values.flatten()
    psi_imag_rect = rec_barrier['psi_imag'].values.flatten()
    psi_real_gauss = gauss_barrier['psi_real'].values.flatten()
    psi_imag_gauss = gauss_barrier['psi_imag'].values.flatten()
    
    print(f"  Rectangular: {len(psi_real_rect):,} points")
    print(f"  Gaussian:    {len(psi_real_gauss):,} points")
    
    # Sample for visualization
    plot_samples = 25000
    print(f"\n  Sampling {plot_samples:,} points for visualization...")
    
    psi_real_rect_plot, psi_imag_rect_plot = stratified_sample_2d(
        psi_real_rect, psi_imag_rect, plot_samples, n_strata=15
    )
    psi_real_gauss_plot, psi_imag_gauss_plot = stratified_sample_2d(
        psi_real_gauss, psi_imag_gauss, plot_samples, n_strata=15
    )
    
    # For statistics, use larger sample
    stat_samples = 500000
    if len(psi_real_rect) > stat_samples:
        print(f"  Sampling {stat_samples:,} points for statistics...")
        psi_real_rect_stat, psi_imag_rect_stat = stratified_sample_2d(
            psi_real_rect, psi_imag_rect, stat_samples, n_strata=20
        )
        psi_real_gauss_stat, psi_imag_gauss_stat = stratified_sample_2d(
            psi_real_gauss, psi_imag_gauss, stat_samples, n_strata=20
        )
    else:
        psi_real_rect_stat = psi_real_rect
        psi_imag_rect_stat = psi_imag_rect
        psi_real_gauss_stat = psi_real_gauss
        psi_imag_gauss_stat = psi_imag_gauss
    
    print("✓ Sampling complete")
    
    # Color scheme
    colors = {
        'rect': '#1E88E5',      # Deep blue
        'gauss': '#E53935',     # Vibrant red
    }
    
    # Set style
    print("\n[3/5] Creating phase space visualization...")
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
        'font.size': 13,
        'axes.labelsize': 15,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'figure.dpi': 600,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'axes.linewidth': 1.5,
        'text.usetex': False
    })
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    
    # Panel (a): Rectangular barrier - BLUE
    ax1 = axes[0]
    
    ax1.scatter(psi_real_rect_plot, psi_imag_rect_plot,
                s=4.0, alpha=0.45, c=colors['rect'], 
                edgecolors='none', rasterized=True)
    
    ax1.set_xlabel(r'$\mathrm{Re}(\psi)$ [nm$^{-1/2}$]', fontsize=15, weight='600')
    ax1.set_ylabel(r'$\mathrm{Im}(\psi)$ [nm$^{-1/2}$]', fontsize=15, weight='600')
    ax1.text(0.03, 0.97, '(a)', transform=ax1.transAxes,
             fontsize=18, fontweight='bold', verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='black', linewidth=2, alpha=0.95))
    
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, color='gray')
    ax1.set_aspect('equal', adjustable='box')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1.0, alpha=0.5)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1.0, alpha=0.5)
    ax1.spines['top'].set_visible(True)
    ax1.spines['right'].set_visible(True)
    
    # Panel (b): Gaussian barrier - RED
    ax2 = axes[1]
    
    ax2.scatter(psi_real_gauss_plot, psi_imag_gauss_plot,
                s=4.0, alpha=0.45, c=colors['gauss'],
                edgecolors='none', rasterized=True)
    
    ax2.set_xlabel(r'$\mathrm{Re}(\psi)$ [nm$^{-1/2}$]', fontsize=15, weight='600')
    ax2.set_ylabel(r'$\mathrm{Im}(\psi)$ [nm$^{-1/2}$]', fontsize=15, weight='600')
    ax2.text(0.03, 0.97, '(b)', transform=ax2.transAxes,
             fontsize=18, fontweight='bold', verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='black', linewidth=2, alpha=0.95))
    
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, color='gray')
    ax2.set_aspect('equal', adjustable='box')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.0, alpha=0.5)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1.0, alpha=0.5)
    ax2.spines['top'].set_visible(True)
    ax2.spines['right'].set_visible(True)
    
    plt.tight_layout()
    
    # Save figure
    fig_path = "../figs/phase_space_comparison.png"
    plt.savefig(fig_path, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"✓ Figure saved: {fig_path}")
    plt.close()
    
    # ============================================================
    # STATISTICAL ANALYSIS
    # ============================================================
    
    print("\n[4/5] Computing statistics...")
    
    results = []
    results.append("=" * 80)
    results.append("PHASE SPACE STATISTICAL ANALYSIS")
    results.append("Complex Wavefunction: Re(ψ) vs Im(ψ)")
    results.append("=" * 80)
    results.append(f"\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results.append("\n" + "=" * 80)
    results.append("DATA INFORMATION")
    results.append("=" * 80)
    results.append(f"\nOriginal dataset sizes:")
    results.append(f"  Rectangular: {len(psi_real_rect):,} points")
    results.append(f"  Gaussian:    {len(psi_real_gauss):,} points")
    results.append(f"\nStatistical analysis sample sizes:")
    results.append(f"  Rectangular: {len(psi_real_rect_stat):,} points")
    results.append(f"  Gaussian:    {len(psi_imag_gauss_stat):,} points")
    
    # ============================================================
    # RECTANGULAR BARRIER
    # ============================================================
    
    results.append("\n" + "=" * 80)
    results.append("RECTANGULAR BARRIER - PHASE SPACE ANALYSIS")
    results.append("=" * 80)
    
    results.append("\n--- REAL COMPONENT Re(ψ) ---")
    results.append(f"  Mean:     {np.mean(psi_real_rect_stat):.3e} nm⁻¹/²")
    results.append(f"  Std Dev:  {np.std(psi_real_rect_stat):.3e} nm⁻¹/²")
    results.append(f"  Median:   {np.median(psi_real_rect_stat):.3e} nm⁻¹/²")
    results.append(f"  Min:      {np.min(psi_real_rect_stat):.3e} nm⁻¹/²")
    results.append(f"  Max:      {np.max(psi_real_rect_stat):.3e} nm⁻¹/²")
    results.append(f"  Skewness: {stats.skew(psi_real_rect_stat):.3f}")
    results.append(f"  Kurtosis: {stats.kurtosis(psi_real_rect_stat):.3f}")
    
    results.append("\n--- IMAGINARY COMPONENT Im(ψ) ---")
    results.append(f"  Mean:     {np.mean(psi_imag_rect_stat):.3e} nm⁻¹/²")
    results.append(f"  Std Dev:  {np.std(psi_imag_rect_stat):.3e} nm⁻¹/²")
    results.append(f"  Median:   {np.median(psi_imag_rect_stat):.3e} nm⁻¹/²")
    results.append(f"  Min:      {np.min(psi_imag_rect_stat):.3e} nm⁻¹/²")
    results.append(f"  Max:      {np.max(psi_imag_rect_stat):.3e} nm⁻¹/²")
    results.append(f"  Skewness: {stats.skew(psi_imag_rect_stat):.3f}")
    results.append(f"  Kurtosis: {stats.kurtosis(psi_imag_rect_stat):.3f}")
    
    r_rect, r_stats_rect = radial_distribution_stats(psi_real_rect_stat, psi_imag_rect_stat)
    results.append("\n--- RADIAL DISTRIBUTION |ψ| ---")
    results.append(f"  Mean |ψ|:     {r_stats_rect['mean']:.3e} nm⁻¹/²")
    results.append(f"  Std Dev:      {r_stats_rect['std']:.3e} nm⁻¹/²")
    results.append(f"  Median:       {r_stats_rect['median']:.3e} nm⁻¹/²")
    results.append(f"  IQR:          {r_stats_rect['iqr']:.3e} nm⁻¹/²")
    results.append(f"  Skewness:     {r_stats_rect['skewness']:.3f}")
    results.append(f"  Kurtosis:     {r_stats_rect['kurtosis']:.3f}")
    
    theta_rect, theta_stats_rect = angular_distribution_stats(psi_real_rect_stat, psi_imag_rect_stat)
    results.append("\n--- ANGULAR DISTRIBUTION arg(ψ) ---")
    results.append(f"  Circular mean:      {theta_stats_rect['circular_mean']:.3f} rad")
    results.append(f"  Circular variance:  {theta_stats_rect['circular_variance']:.3f}")
    results.append(f"  Concentration (R):  {theta_stats_rect['concentration']:.3f}")
    results.append(f"  Angular std dev:    {theta_stats_rect['std_dev']:.3f} rad")
    results.append(f"  Angular entropy:    {theta_stats_rect['entropy']:.3f} bits")
    
    H_ps_rect = phase_space_entropy(psi_real_rect_stat, psi_imag_rect_stat, bins=60)
    results.append("\n--- PHASE SPACE ENTROPY ---")
    results.append(f"  2D Shannon Entropy: {H_ps_rect:.3f} bits")
    
    mi_rect, H_re_rect, H_im_rect, H_joint_rect = mutual_information_phase_space(
        psi_real_rect_stat, psi_imag_rect_stat, bins=60
    )
    results.append("\n--- MUTUAL INFORMATION ---")
    results.append(f"  I(Re;Im) = {mi_rect:.3f} bits")
    results.append(f"  H(Re) = {H_re_rect:.3f} bits")
    results.append(f"  H(Im) = {H_im_rect:.3f} bits")
    results.append(f"  H(Re,Im) = {H_joint_rect:.3f} bits")
    results.append(f"  Normalized MI: {mi_rect / min(H_re_rect, H_im_rect):.3f}")
    
    pci_rect = phase_coherence_index(psi_real_rect_stat, psi_imag_rect_stat)
    results.append("\n--- PHASE COHERENCE INDEX ---")
    results.append(f"  PCI = |⟨ψ⟩| / ⟨|ψ|⟩ = {pci_rect:.3f}")
    
    ar_rect, eigenvals_rect = anisotropy_ratio(psi_real_rect_stat, psi_imag_rect_stat)
    results.append("\n--- ANISOTROPY ANALYSIS ---")
    results.append(f"  Anisotropy Ratio: {ar_rect:.3f}")
    results.append(f"  Eigenvalues: λ₁ = {eigenvals_rect[1]:.3e}, λ₂ = {eigenvals_rect[0]:.3e}")
    results.append(f"  Aspect ratio: {np.sqrt(eigenvals_rect[1]/eigenvals_rect[0]):.3f}")
    
    e_rect, a_rect, b_rect = ellipticity_measure(psi_real_rect_stat, psi_imag_rect_stat)
    results.append("\n--- ELLIPTICITY MEASURE ---")
    results.append(f"  Ellipticity: e = {e_rect:.3f}")
    results.append(f"  Semi-major axis: a = {a_rect:.3e} nm⁻¹/²")
    results.append(f"  Semi-minor axis: b = {b_rect:.3e} nm⁻¹/²")
    
    area_rect = phase_space_area(psi_real_rect_stat, psi_imag_rect_stat, quantile=0.95)
    results.append("\n--- PHASE SPACE AREA (95% Quantile) ---")
    results.append(f"  Effective area: {area_rect:.3e} nm⁻¹")
    
    m20_rect = phase_space_moment(psi_real_rect_stat, psi_imag_rect_stat, 2, 0)
    m02_rect = phase_space_moment(psi_real_rect_stat, psi_imag_rect_stat, 0, 2)
    m11_rect = phase_space_moment(psi_real_rect_stat, psi_imag_rect_stat, 1, 1)
    results.append("\n--- PHASE SPACE MOMENTS ---")
    results.append(f"  ⟨Re²⟩ = {m20_rect:.3e} nm⁻¹")
    results.append(f"  ⟨Im²⟩ = {m02_rect:.3e} nm⁻¹")
    results.append(f"  ⟨Re·Im⟩ = {m11_rect:.3e} nm⁻¹")
    
    pearson_r_rect, pearson_p_rect = stats.pearsonr(psi_real_rect_stat, psi_imag_rect_stat)
    results.append("\n--- LINEAR CORRELATION ---")
    results.append(f"  Pearson r: {pearson_r_rect:.3f}")
    results.append(f"  P-value: {pearson_p_rect:.3e}")
    
    # ============================================================
    # GAUSSIAN BARRIER
    # ============================================================
    
    results.append("\n" + "=" * 80)
    results.append("GAUSSIAN BARRIER - PHASE SPACE ANALYSIS")
    results.append("=" * 80)
    
    results.append("\n--- REAL COMPONENT Re(ψ) ---")
    results.append(f"  Mean:     {np.mean(psi_real_gauss_stat):.3e} nm⁻¹/²")
    results.append(f"  Std Dev:  {np.std(psi_real_gauss_stat):.3e} nm⁻¹/²")
    results.append(f"  Median:   {np.median(psi_real_gauss_stat):.3e} nm⁻¹/²")
    results.append(f"  Min:      {np.min(psi_real_gauss_stat):.3e} nm⁻¹/²")
    results.append(f"  Max:      {np.max(psi_real_gauss_stat):.3e} nm⁻¹/²")
    results.append(f"  Skewness: {stats.skew(psi_real_gauss_stat):.3f}")
    results.append(f"  Kurtosis: {stats.kurtosis(psi_real_gauss_stat):.3f}")
    
    results.append("\n--- IMAGINARY COMPONENT Im(ψ) ---")
    results.append(f"  Mean:     {np.mean(psi_imag_gauss_stat):.3e} nm⁻¹/²")
    results.append(f"  Std Dev:  {np.std(psi_imag_gauss_stat):.3e} nm⁻¹/²")
    results.append(f"  Median:   {np.median(psi_imag_gauss_stat):.3e} nm⁻¹/²")
    results.append(f"  Min:      {np.min(psi_imag_gauss_stat):.3e} nm⁻¹/²")
    results.append(f"  Max:      {np.max(psi_imag_gauss_stat):.3e} nm⁻¹/²")
    results.append(f"  Skewness: {stats.skew(psi_imag_gauss_stat):.3f}")
    results.append(f"  Kurtosis: {stats.kurtosis(psi_imag_gauss_stat):.3f}")
    
    r_gauss, r_stats_gauss = radial_distribution_stats(psi_real_gauss_stat, psi_imag_gauss_stat)
    results.append("\n--- RADIAL DISTRIBUTION |ψ| ---")
    results.append(f"  Mean |ψ|:     {r_stats_gauss['mean']:.3e} nm⁻¹/²")
    results.append(f"  Std Dev:      {r_stats_gauss['std']:.3e} nm⁻¹/²")
    results.append(f"  Median:       {r_stats_gauss['median']:.3e} nm⁻¹/²")
    results.append(f"  IQR:          {r_stats_gauss['iqr']:.3e} nm⁻¹/²")
    results.append(f"  Skewness:     {r_stats_gauss['skewness']:.3f}")
    results.append(f"  Kurtosis:     {r_stats_gauss['kurtosis']:.3f}")
    
    theta_gauss, theta_stats_gauss = angular_distribution_stats(psi_real_gauss_stat, psi_imag_gauss_stat)
    results.append("\n--- ANGULAR DISTRIBUTION arg(ψ) ---")
    results.append(f"  Circular mean:      {theta_stats_gauss['circular_mean']:.3f} rad")
    results.append(f"  Circular variance:  {theta_stats_gauss['circular_variance']:.3f}")
    results.append(f"  Concentration (R):  {theta_stats_gauss['concentration']:.3f}")
    results.append(f"  Angular std dev:    {theta_stats_gauss['std_dev']:.3f} rad")
    results.append(f"  Angular entropy:    {theta_stats_gauss['entropy']:.3f} bits")
    
    H_ps_gauss = phase_space_entropy(psi_real_gauss_stat, psi_imag_gauss_stat, bins=60)
    results.append("\n--- PHASE SPACE ENTROPY ---")
    results.append(f"  2D Shannon Entropy: {H_ps_gauss:.3f} bits")
    
    mi_gauss, H_re_gauss, H_im_gauss, H_joint_gauss = mutual_information_phase_space(
        psi_real_gauss_stat, psi_imag_gauss_stat, bins=60
    )
    results.append("\n--- MUTUAL INFORMATION ---")
    results.append(f"  I(Re;Im) = {mi_gauss:.3f} bits")
    results.append(f"  H(Re) = {H_re_gauss:.3f} bits")
    results.append(f"  H(Im) = {H_im_gauss:.3f} bits")
    results.append(f"  H(Re,Im) = {H_joint_gauss:.3f} bits")
    results.append(f"  Normalized MI: {mi_gauss / min(H_re_gauss, H_im_gauss):.3f}")
    
    pci_gauss = phase_coherence_index(psi_real_gauss_stat, psi_imag_gauss_stat)
    results.append("\n--- PHASE COHERENCE INDEX ---")
    results.append(f"  PCI = |⟨ψ⟩| / ⟨|ψ|⟩ = {pci_gauss:.3f}")
    
    ar_gauss, eigenvals_gauss = anisotropy_ratio(psi_real_gauss_stat, psi_imag_gauss_stat)
    results.append("\n--- ANISOTROPY ANALYSIS ---")
    results.append(f"  Anisotropy Ratio: {ar_gauss:.3f}")
    results.append(f"  Eigenvalues: λ₁ = {eigenvals_gauss[1]:.3e}, λ₂ = {eigenvals_gauss[0]:.3e}")
    results.append(f"  Aspect ratio: {np.sqrt(eigenvals_gauss[1]/eigenvals_gauss[0]):.3f}")
    
    e_gauss, a_gauss, b_gauss = ellipticity_measure(psi_real_gauss_stat, psi_imag_gauss_stat)
    results.append("\n--- ELLIPTICITY MEASURE ---")
    results.append(f"  Ellipticity: e = {e_gauss:.3f}")
    results.append(f"  Semi-major axis: a = {a_gauss:.3e} nm⁻¹/²")
    results.append(f"  Semi-minor axis: b = {b_gauss:.3e} nm⁻¹/²")
    
    area_gauss = phase_space_area(psi_real_gauss_stat, psi_imag_gauss_stat, quantile=0.95)
    results.append("\n--- PHASE SPACE AREA (95% Quantile) ---")
    results.append(f"  Effective area: {area_gauss:.3e} nm⁻¹")
    
    m20_gauss = phase_space_moment(psi_real_gauss_stat, psi_imag_gauss_stat, 2, 0)
    m02_gauss = phase_space_moment(psi_real_gauss_stat, psi_imag_gauss_stat, 0, 2)
    m11_gauss = phase_space_moment(psi_real_gauss_stat, psi_imag_gauss_stat, 1, 1)
    results.append("\n--- PHASE SPACE MOMENTS ---")
    results.append(f"  ⟨Re²⟩ = {m20_gauss:.3e} nm⁻¹")
    results.append(f"  ⟨Im²⟩ = {m02_gauss:.3e} nm⁻¹")
    results.append(f"  ⟨Re·Im⟩ = {m11_gauss:.3e} nm⁻¹")
    
    pearson_r_gauss, pearson_p_gauss = stats.pearsonr(psi_real_gauss_stat, psi_imag_gauss_stat)
    results.append("\n--- LINEAR CORRELATION ---")
    results.append(f"  Pearson r: {pearson_r_gauss:.3f}")
    results.append(f"  P-value: {pearson_p_gauss:.3e}")
    
    # ============================================================
    # COMPARATIVE ANALYSIS
    # ============================================================
    
    results.append("\n" + "=" * 80)
    results.append("COMPARATIVE PHASE SPACE ANALYSIS")
    results.append("=" * 80)
    
    results.append("\n--- ENTROPY COMPARISON ---")
    results.append(f"  2D Entropy (Rectangular): {H_ps_rect:.3f} bits")
    results.append(f"  2D Entropy (Gaussian):    {H_ps_gauss:.3f} bits")
    results.append(f"  Difference:               {abs(H_ps_rect - H_ps_gauss):.3f} bits")
    
    results.append("\n--- MUTUAL INFORMATION COMPARISON ---")
    results.append(f"  MI (Rectangular): {mi_rect:.3f} bits")
    results.append(f"  MI (Gaussian):    {mi_gauss:.3f} bits")
    results.append(f"  Difference:       {abs(mi_rect - mi_gauss):.3f} bits")
    
    results.append("\n--- PHASE COHERENCE COMPARISON ---")
    results.append(f"  PCI (Rectangular): {pci_rect:.3f}")
    results.append(f"  PCI (Gaussian):    {pci_gauss:.3f}")
    results.append(f"  Difference:        {abs(pci_rect - pci_gauss):.3f}")
    
    results.append("\n--- ANISOTROPY COMPARISON ---")
    results.append(f"  Anisotropy (Rectangular): {ar_rect:.3f}")
    results.append(f"  Anisotropy (Gaussian):    {ar_gauss:.3f}")
    results.append(f"  Difference:               {abs(ar_rect - ar_gauss):.3f}")
    
    results.append("\n--- ELLIPTICITY COMPARISON ---")
    results.append(f"  Ellipticity (Rectangular): {e_rect:.3f}")
    results.append(f"  Ellipticity (Gaussian):    {e_gauss:.3f}")
    results.append(f"  Difference:                {abs(e_rect - e_gauss):.3f}")
    
    results.append("\n--- PHASE SPACE AREA COMPARISON ---")
    results.append(f"  Area (Rectangular): {area_rect:.3e} nm⁻¹")
    results.append(f"  Area (Gaussian):    {area_gauss:.3e} nm⁻¹")
    results.append(f"  Ratio (Rect/Gauss): {area_rect/area_gauss:.3f}")
    
    results.append("\n--- ANGULAR CONCENTRATION COMPARISON ---")
    results.append(f"  Concentration (Rectangular): {theta_stats_rect['concentration']:.3f}")
    results.append(f"  Concentration (Gaussian):    {theta_stats_gauss['concentration']:.3f}")
    results.append(f"  Difference:                  {abs(theta_stats_rect['concentration'] - theta_stats_gauss['concentration']):.3f}")
    
    # Statistical tests
    results.append("\n--- STATISTICAL TESTS ---")
    ks_stat_r, ks_pval_r = stats.ks_2samp(r_rect, r_gauss)
    results.append(f"\n  Kolmogorov-Smirnov Test (|ψ|):")
    results.append(f"    KS Statistic: {ks_stat_r:.3f}")
    results.append(f"    P-value:      {ks_pval_r:.3e}")
    
    results.append(f"\n  Angular Variance Comparison:")
    results.append(f"    Circular variance (Rectangular): {theta_stats_rect['circular_variance']:.3f}")
    results.append(f"    Circular variance (Gaussian):    {theta_stats_gauss['circular_variance']:.3f}")
    results.append(f"    Difference: {abs(theta_stats_rect['circular_variance'] - theta_stats_gauss['circular_variance']):.3f}")
    
    results.append("\n" + "=" * 80)
    results.append("SUMMARY")
    results.append("=" * 80)
    
    results.append(f"\nKey Findings:")
    results.append(f"  • Entropy difference: {abs(H_ps_rect - H_ps_gauss):.3f} bits")
    results.append(f"  • Phase space area ratio: {area_rect/area_gauss:.3f}")
    results.append(f"  • Phase coherence (Rect): {pci_rect:.3f}")
    results.append(f"  • Phase coherence (Gauss): {pci_gauss:.3f}")
    results.append(f"  • Anisotropy (Rect): {ar_rect:.3f}")
    results.append(f"  • Anisotropy (Gauss): {ar_gauss:.3f}")
    results.append(f"  • Radial distribution KS p-value: {ks_pval_r:.3e}")
    
    results.append("\n" + "=" * 80)
    results.append("END OF ANALYSIS")
    results.append("=" * 80)
    
    # Write results
    print("\n[5/5] Saving statistics...")
    output_file = "../stats/phase_space_stats.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results))
    
    print(f"✓ Statistics saved: {output_file}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Figure:     ../figs/phase_space_comparison.png (600 DPI)")
    print(f"Statistics: ../stats/phase_space_stats.txt")
    print(f"\nKey Results:")
    print(f"  • Entropy difference: {abs(H_ps_rect - H_ps_gauss):.3f} bits")
    print(f"  • Area ratio: {area_rect/area_gauss:.3f}")
    print(f"  • PCI (Rect): {pci_rect:.3f}, (Gauss): {pci_gauss:.3f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
