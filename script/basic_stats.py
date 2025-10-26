#!/usr/bin/env python
"""
Optimized Quantum Tunneling Probability Distribution Analysis
with Robust Entropy Measures and Efficient Sampling
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde
from scipy.optimize import minimize_scalar
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def optimal_bins(data, method='auto'):
    """
    Calculate optimal number of bins using multiple robust methods.
    Returns dict with different bin estimates.
    """
    n = len(data)
    data_range = np.ptp(data)
    
    bins_dict = {}
    
    # Sturges' rule
    bins_dict['sturges'] = int(np.ceil(np.log2(n) + 1))
    
    # Scott's rule
    h_scott = 3.5 * np.std(data) / (n ** (1/3))
    bins_dict['scott'] = int(np.ceil(data_range / h_scott)) if h_scott > 0 else 10
    
    # Freedman-Diaconis rule (robust to outliers)
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    h_fd = 2 * iqr / (n ** (1/3))
    bins_dict['freedman_diaconis'] = int(np.ceil(data_range / h_fd)) if h_fd > 0 else 10
    
    # Rice rule
    bins_dict['rice'] = int(np.ceil(2 * n ** (1/3)))
    
    # Square root rule
    bins_dict['sqrt'] = int(np.ceil(np.sqrt(n)))
    
    # Doane's formula (extension of Sturges for non-normal data)
    g1 = stats.skew(data)
    sigma_g1 = np.sqrt((6*(n-2)) / ((n+1)*(n+3)))
    bins_dict['doane'] = int(np.ceil(1 + np.log2(n) + np.log2(1 + abs(g1)/sigma_g1)))
    
    # Clip all to reasonable range
    for key in bins_dict:
        bins_dict[key] = np.clip(bins_dict[key], 5, 500)
    
    # Consensus estimate (robust median)
    bins_dict['consensus'] = int(np.median(list(bins_dict.values())))
    
    if method == 'auto':
        return bins_dict['consensus'], bins_dict
    else:
        return bins_dict.get(method, bins_dict['consensus']), bins_dict


def shannon_entropy(data, bins='auto'):
    """
    Calculate Shannon entropy with optimal binning.
    H = -∑ p(x) * log2(p(x))
    """
    if isinstance(bins, str) and bins == 'auto':
        bins, _ = optimal_bins(data)
    
    hist, _ = np.histogram(data, bins=bins, density=True)
    hist = hist[hist > 0]  # Remove zero bins
    
    # Normalize to probability distribution
    prob = hist / np.sum(hist)
    
    # Calculate entropy
    entropy = -np.sum(prob * np.log2(prob))
    return entropy, bins


def cross_entropy(p_data, q_data, bins='auto'):
    """
    Calculate cross-entropy H(P,Q) = -∑ p(x) * log2(q(x))
    Measures dissimilarity between two distributions.
    """
    if isinstance(bins, str) and bins == 'auto':
        # Use common binning for fair comparison
        bins_p, _ = optimal_bins(p_data)
        bins_q, _ = optimal_bins(q_data)
        bins = int(np.mean([bins_p, bins_q]))
    
    # Create common bin edges
    bin_edges = np.linspace(
        min(p_data.min(), q_data.min()),
        max(p_data.max(), q_data.max()),
        bins + 1
    )
    
    hist_p, _ = np.histogram(p_data, bins=bin_edges, density=True)
    hist_q, _ = np.histogram(q_data, bins=bin_edges, density=True)
    
    # Normalize
    prob_p = hist_p / (np.sum(hist_p) + 1e-12)
    prob_q = hist_q / (np.sum(hist_q) + 1e-12)
    
    # Add small epsilon to avoid log(0)
    prob_q = np.maximum(prob_q, 1e-12)
    
    # Calculate cross-entropy
    ce = -np.sum(prob_p * np.log2(prob_q))
    return ce, bins


def kl_divergence(p_data, q_data, bins='auto'):
    """
    Calculate Kullback-Leibler divergence: KL(P||Q) = ∑ p(x) * log2(p(x)/q(x))
    Measures how P diverges from Q (asymmetric).
    """
    if isinstance(bins, str) and bins == 'auto':
        bins_p, _ = optimal_bins(p_data)
        bins_q, _ = optimal_bins(q_data)
        bins = int(np.mean([bins_p, bins_q]))
    
    bin_edges = np.linspace(
        min(p_data.min(), q_data.min()),
        max(p_data.max(), q_data.max()),
        bins + 1
    )
    
    hist_p, _ = np.histogram(p_data, bins=bin_edges, density=True)
    hist_q, _ = np.histogram(q_data, bins=bin_edges, density=True)
    
    prob_p = hist_p / (np.sum(hist_p) + 1e-12)
    prob_q = hist_q / (np.sum(hist_q) + 1e-12)
    
    # Add epsilon to avoid division by zero
    prob_p = np.maximum(prob_p, 1e-12)
    prob_q = np.maximum(prob_q, 1e-12)
    
    # KL divergence (only where p > 0)
    mask = prob_p > 1e-10
    kl = np.sum(prob_p[mask] * np.log2(prob_p[mask] / prob_q[mask]))
    
    return kl, bins


def js_divergence(p_data, q_data, bins='auto'):
    """
    Calculate Jensen-Shannon divergence (symmetric version of KL).
    JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), where M = 0.5(P+Q)
    Range: [0, 1] (in bits)
    """
    if isinstance(bins, str) and bins == 'auto':
        bins_p, _ = optimal_bins(p_data)
        bins_q, _ = optimal_bins(q_data)
        bins = int(np.mean([bins_p, bins_q]))
    
    bin_edges = np.linspace(
        min(p_data.min(), q_data.min()),
        max(p_data.max(), q_data.max()),
        bins + 1
    )
    
    hist_p, _ = np.histogram(p_data, bins=bin_edges, density=True)
    hist_q, _ = np.histogram(q_data, bins=bin_edges, density=True)
    
    prob_p = hist_p / (np.sum(hist_p) + 1e-12)
    prob_q = hist_q / (np.sum(hist_q) + 1e-12)
    
    # Mixture distribution
    prob_m = 0.5 * (prob_p + prob_q)
    
    prob_p = np.maximum(prob_p, 1e-12)
    prob_q = np.maximum(prob_q, 1e-12)
    prob_m = np.maximum(prob_m, 1e-12)
    
    # Calculate KL divergences
    mask_p = prob_p > 1e-10
    mask_q = prob_q > 1e-10
    
    kl_pm = np.sum(prob_p[mask_p] * np.log2(prob_p[mask_p] / prob_m[mask_p]))
    kl_qm = np.sum(prob_q[mask_q] * np.log2(prob_q[mask_q] / prob_m[mask_q]))
    
    js = 0.5 * kl_pm + 0.5 * kl_qm
    
    return js, bins


def mutual_information(x_data, y_data, bins='auto'):
    """
    Calculate mutual information I(X;Y) = H(X) + H(Y) - H(X,Y)
    Measures the amount of information shared between two variables.
    """
    if isinstance(bins, str) and bins == 'auto':
        bins_x, _ = optimal_bins(x_data)
        bins_y, _ = optimal_bins(y_data)
        bins = int(np.mean([bins_x, bins_y]))
    
    # Individual entropies
    hx, _ = shannon_entropy(x_data, bins)
    hy, _ = shannon_entropy(y_data, bins)
    
    # Joint histogram
    hist_2d, _, _ = np.histogram2d(x_data, y_data, bins=bins)
    prob_2d = hist_2d / np.sum(hist_2d)
    prob_2d = prob_2d[prob_2d > 0]
    
    # Joint entropy
    h_xy = -np.sum(prob_2d * np.log2(prob_2d))
    
    # Mutual information
    mi = hx + hy - h_xy
    
    return mi, bins


def stratified_sample(data, n_samples, n_strata=10):
    """
    Perform stratified sampling to preserve distribution characteristics.
    Divides data into strata and samples proportionally from each.
    """
    if n_samples >= len(data):
        return data
    
    # Sort data
    sorted_data = np.sort(data)
    n = len(sorted_data)
    
    # Calculate samples per stratum
    samples_per_stratum = n_samples // n_strata
    remainder = n_samples % n_strata
    
    sampled_indices = []
    
    for i in range(n_strata):
        # Define stratum boundaries
        start_idx = int(i * n / n_strata)
        end_idx = int((i + 1) * n / n_strata)
        
        # Number of samples for this stratum
        n_stratum_samples = samples_per_stratum + (1 if i < remainder else 0)
        
        # Random sample from this stratum
        stratum_indices = np.arange(start_idx, end_idx)
        if len(stratum_indices) > 0:
            sampled = np.random.choice(stratum_indices, 
                                      size=min(n_stratum_samples, len(stratum_indices)),
                                      replace=False)
            sampled_indices.extend(sampled)
    
    return sorted_data[sampled_indices]


def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("\n" + "=" * 80)
    print("QUANTUM TUNNELING PROBABILITY DISTRIBUTION ANALYSIS")
    print("Optimized with Sampling and Entropy Analysis")
    print("=" * 80)
    
    # Load data
    print("\n[1/6] Loading NetCDF datasets...")
    rec_barrier = xr.open_dataset("../raw_data/case1_rectangular_barrier.nc")
    gauss_barrier = xr.open_dataset("../raw_data/case2_gaussian_barrier.nc")
    print("✓ Datasets loaded successfully")
    
    # Create output directories
    Path("../figs").mkdir(parents=True, exist_ok=True)
    Path("../stats").mkdir(parents=True, exist_ok=True)
    
    # Extract and filter probability data
    print("\n[2/6] Processing probability distributions...")
    prob_rect_full = rec_barrier['probability'].values.flatten()
    prob_gauss_full = gauss_barrier['probability'].values.flatten()
    
    # Remove zeros or very small values
    prob_rect_filtered_full = prob_rect_full[prob_rect_full > 1e-10]
    prob_gauss_filtered_full = prob_gauss_full[prob_gauss_full > 1e-10]
    
    print(f"  Original data points - Rectangular: {len(prob_rect_filtered_full):,}")
    print(f"  Original data points - Gaussian: {len(prob_gauss_filtered_full):,}")
    
    # Efficient stratified sampling (maintain ~100k points for balance of speed and accuracy)
    max_samples = 100000
    
    if len(prob_rect_filtered_full) > max_samples:
        print(f"\n  Applying stratified sampling to {max_samples:,} points per dataset...")
        prob_rect_filtered = stratified_sample(prob_rect_filtered_full, max_samples, n_strata=20)
        prob_gauss_filtered = stratified_sample(prob_gauss_filtered_full, max_samples, n_strata=20)
        print("✓ Sampling complete - distribution characteristics preserved")
    else:
        prob_rect_filtered = prob_rect_filtered_full
        prob_gauss_filtered = prob_gauss_filtered_full
    
    print(f"  Analysis data points - Rectangular: {len(prob_rect_filtered):,}")
    print(f"  Analysis data points - Gaussian: {len(prob_gauss_filtered):,}")
    
    # Stunning publication color scheme (inspired by Nature/Science journals)
    colors = {
        'rect': '#0A5F8C',          # Deep ocean blue
        'gauss': '#D94E1F',         # Burnt sienna orange
        'rect_fill': '#4A9EC7',     # Sky blue
        'gauss_fill': '#F4A460',    # Sandy brown
        'rect_light': '#85C5E3',    # Light blue
        'gauss_light': '#FFB884',   # Peach
        'grid': '#E5E5E5',          # Light gray
        'background': '#FAFAFA',     # Off-white
        'text': '#2C2C2C'           # Charcoal
    }
    
    # Set publication-quality style
    print("\n[3/6] Configuring publication-quality graphics...")
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
        'font.size': 11,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.dpi': 600,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'axes.linewidth': 1.2,
        'grid.linewidth': 0.6,
        'lines.linewidth': 2.0,
        'patch.linewidth': 1.2,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.minor.width': 0.8,
        'ytick.minor.width': 0.8,
        'text.usetex': False,
        'axes.facecolor': colors['background'],
        'figure.facecolor': 'white',
        'axes.edgecolor': colors['text'],
        'xtick.color': colors['text'],
        'ytick.color': colors['text'],
        'axes.labelcolor': colors['text']
    })
    
    # Create visualization
    print("\n[4/6] Generating publication-quality visualization...")
    fig, axes = plt.subplots(2, 1, figsize=(10, 9))
    
    # --- Plot (a): KDE Comparison with Normalized Density ---
    ax1 = axes[0]
    
    # Sample for KDE (further reduce for speed)
    kde_samples = 50000
    rect_kde_sample = stratified_sample(prob_rect_filtered, min(kde_samples, len(prob_rect_filtered)))
    gauss_kde_sample = stratified_sample(prob_gauss_filtered, min(kde_samples, len(prob_gauss_filtered)))
    
    # Rectangular barrier KDE
    kde_rect = gaussian_kde(rect_kde_sample, bw_method='scott')
    x_rect = np.linspace(prob_rect_filtered.min(), prob_rect_filtered.max(), 2000)
    kde_rect_vals = kde_rect(x_rect)
    kde_rect_norm = (kde_rect_vals - kde_rect_vals.min()) / (kde_rect_vals.max() - kde_rect_vals.min())
    
    # Plot with gradient effect
    ax1.plot(x_rect, kde_rect_norm, color=colors['rect'], linewidth=2.5, 
             label='Rectangular Barrier', alpha=0.9, zorder=3)
    ax1.fill_between(x_rect, 0, kde_rect_norm, color=colors['rect_fill'], 
                      alpha=0.25, zorder=1)
    
    # Gaussian barrier KDE
    kde_gauss = gaussian_kde(gauss_kde_sample, bw_method='scott')
    x_gauss = np.linspace(prob_gauss_filtered.min(), prob_gauss_filtered.max(), 2000)
    kde_gauss_vals = kde_gauss(x_gauss)
    kde_gauss_norm = (kde_gauss_vals - kde_gauss_vals.min()) / (kde_gauss_vals.max() - kde_gauss_vals.min())
    
    ax1.plot(x_gauss, kde_gauss_norm, color=colors['gauss'], linewidth=2.5,
             label='Gaussian Barrier', alpha=0.9, zorder=3)
    ax1.fill_between(x_gauss, 0, kde_gauss_norm, color=colors['gauss_fill'],
                      alpha=0.25, zorder=1)
    
    ax1.set_xlabel(r'Probability $|\psi|^2$ [nm$^{-1}$]', fontweight='600', fontsize=13)
    ax1.set_ylabel('Normalized Density', fontweight='600', fontsize=13)
    ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes,
             fontsize=16, fontweight='bold', verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                      edgecolor=colors['text'], linewidth=1.5, alpha=0.9))
    ax1.grid(True, alpha=0.35, color=colors['grid'], linestyle='-', linewidth=0.6)
    ax1.legend(frameon=True, fancybox=False, shadow=False, loc='upper right',
               framealpha=0.95, edgecolor=colors['text'], facecolor='white')
    ax1.set_ylim(-0.02, 1.08)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # --- Plot (b): Box Plot Comparison ---
    ax2 = axes[1]
    
    # Sample for boxplot (violin plots work better with ~10k points)
    box_samples = 20000
    rect_box_sample = stratified_sample(prob_rect_filtered, min(box_samples, len(prob_rect_filtered)))
    gauss_box_sample = stratified_sample(prob_gauss_filtered, min(box_samples, len(prob_gauss_filtered)))
    
    bp = ax2.boxplot([rect_box_sample, gauss_box_sample],
                      positions=[1, 2],
                      widths=0.5,
                      patch_artist=True,
                      labels=['Rectangular', 'Gaussian'],
                      boxprops=dict(linewidth=1.8),
                      whiskerprops=dict(linewidth=1.8),
                      capprops=dict(linewidth=1.8),
                      medianprops=dict(color='#C41E3A', linewidth=2.5, zorder=4),
                      flierprops=dict(marker='o', markersize=3, alpha=0.3,
                                     markeredgewidth=0))
    
    # Enhanced box coloring with gradients
    bp['boxes'][0].set_facecolor(colors['rect_light'])
    bp['boxes'][0].set_edgecolor(colors['rect'])
    bp['boxes'][0].set_linewidth(2.0)
    bp['boxes'][1].set_facecolor(colors['gauss_light'])
    bp['boxes'][1].set_edgecolor(colors['gauss'])
    bp['boxes'][1].set_linewidth(2.0)
    
    for i in range(4):
        if i < 2:
            bp['whiskers'][i].set_color(colors['rect'])
            bp['caps'][i].set_color(colors['rect'])
        else:
            bp['whiskers'][i].set_color(colors['gauss'])
            bp['caps'][i].set_color(colors['gauss'])
    
    bp['fliers'][0].set_markerfacecolor(colors['rect'])
    bp['fliers'][0].set_markeredgecolor(colors['rect'])
    bp['fliers'][1].set_markerfacecolor(colors['gauss'])
    bp['fliers'][1].set_markeredgecolor(colors['gauss'])
    
    ax2.set_ylabel(r'Probability $|\psi|^2$ [nm$^{-1}$]', fontweight='600', fontsize=13)
    ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes,
             fontsize=16, fontweight='bold', verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor=colors['text'], linewidth=1.5, alpha=0.9))
    ax2.grid(True, alpha=0.35, axis='y', color=colors['grid'], linestyle='-', linewidth=0.6)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(axis='x', labelsize=12)
    
    plt.tight_layout()
    
    # Save high-resolution PNG only
    fig_path = "../figs/probability_comparison.png"
    plt.savefig(fig_path, format='png', dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"✓ High-resolution figure saved: {fig_path}")
    
    plt.close()
    
    # ============================================================
    # STATISTICAL ANALYSIS WITH ENTROPY MEASURES
    # ============================================================
    
    print("\n[5/6] Computing statistical tests and entropy measures...")
    
    results = []
    results.append("=" * 80)
    results.append("STATISTICAL COMPARISON: RECTANGULAR vs GAUSSIAN BARRIER")
    results.append("Probability Distribution Analysis with Entropy Measures")
    results.append("=" * 80)
    results.append(f"\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Data size information
    results.append("\n" + "=" * 80)
    results.append("DATA SAMPLING INFORMATION")
    results.append("=" * 80)
    results.append(f"\nOriginal dataset sizes:")
    results.append(f"  Rectangular: {len(prob_rect_filtered_full):,} points")
    results.append(f"  Gaussian:    {len(prob_gauss_filtered_full):,} points")
    results.append(f"\nAnalysis dataset sizes (stratified sampling):")
    results.append(f"  Rectangular: {len(prob_rect_filtered):,} points")
    results.append(f"  Gaussian:    {len(prob_gauss_filtered):,} points")
    results.append(f"  Sampling preserves distribution characteristics via stratification")
    
    results.append("\n" + "=" * 80)
    results.append("DESCRIPTIVE STATISTICS")
    results.append("=" * 80)
    
    # Descriptive statistics for Rectangular
    results.append("\n--- RECTANGULAR BARRIER ---")
    results.append(f"Sample size: {len(prob_rect_filtered):,}")
    results.append(f"\nProbability |ψ|² Statistics:")
    results.append(f"  Mean:     {np.mean(prob_rect_filtered):.6e} nm⁻¹")
    results.append(f"  Std Dev:  {np.std(prob_rect_filtered):.6e} nm⁻¹")
    results.append(f"  Median:   {np.median(prob_rect_filtered):.6e} nm⁻¹")
    results.append(f"  Min:      {np.min(prob_rect_filtered):.6e} nm⁻¹")
    results.append(f"  Max:      {np.max(prob_rect_filtered):.6e} nm⁻¹")
    results.append(f"  Q1 (25%): {np.percentile(prob_rect_filtered, 25):.6e} nm⁻¹")
    results.append(f"  Q3 (75%): {np.percentile(prob_rect_filtered, 75):.6e} nm⁻¹")
    results.append(f"  IQR:      {np.percentile(prob_rect_filtered, 75) - np.percentile(prob_rect_filtered, 25):.6e} nm⁻¹")
    results.append(f"  Skewness: {stats.skew(prob_rect_filtered):.4f}")
    results.append(f"  Kurtosis: {stats.kurtosis(prob_rect_filtered):.4f}")
    
    # Descriptive statistics for Gaussian
    results.append("\n--- GAUSSIAN BARRIER ---")
    results.append(f"Sample size: {len(prob_gauss_filtered):,}")
    results.append(f"\nProbability |ψ|² Statistics:")
    results.append(f"  Mean:     {np.mean(prob_gauss_filtered):.6e} nm⁻¹")
    results.append(f"  Std Dev:  {np.std(prob_gauss_filtered):.6e} nm⁻¹")
    results.append(f"  Median:   {np.median(prob_gauss_filtered):.6e} nm⁻¹")
    results.append(f"  Min:      {np.min(prob_gauss_filtered):.6e} nm⁻¹")
    results.append(f"  Max:      {np.max(prob_gauss_filtered):.6e} nm⁻¹")
    results.append(f"  Q1 (25%): {np.percentile(prob_gauss_filtered, 25):.6e} nm⁻¹")
    results.append(f"  Q3 (75%): {np.percentile(prob_gauss_filtered, 75):.6e} nm⁻¹")
    results.append(f"  IQR:      {np.percentile(prob_gauss_filtered, 75) - np.percentile(prob_gauss_filtered, 25):.6e} nm⁻¹")
    results.append(f"  Skewness: {stats.skew(prob_gauss_filtered):.4f}")
    results.append(f"  Kurtosis: {stats.kurtosis(prob_gauss_filtered):.4f}")
    
    # ============================================================
    # ENTROPY ANALYSIS WITH OPTIMAL BINNING
    # ============================================================
    
    results.append("\n" + "=" * 80)
    results.append("INFORMATION-THEORETIC MEASURES (ENTROPY ANALYSIS)")
    results.append("=" * 80)
    
    # Calculate optimal bins for both distributions
    results.append("\n--- OPTIMAL BIN CALCULATION ---")
    bins_rect, bins_dict_rect = optimal_bins(prob_rect_filtered, method='auto')
    bins_gauss, bins_dict_gauss = optimal_bins(prob_gauss_filtered, method='auto')
    
    results.append("\nRectangular Barrier - Bin Optimization:")
    results.append(f"  Sturges' rule:        {bins_dict_rect['sturges']}")
    results.append(f"  Scott's rule:         {bins_dict_rect['scott']}")
    results.append(f"  Freedman-Diaconis:    {bins_dict_rect['freedman_diaconis']}")
    results.append(f"  Rice rule:            {bins_dict_rect['rice']}")
    results.append(f"  Square root rule:     {bins_dict_rect['sqrt']}")
    results.append(f"  Doane's formula:      {bins_dict_rect['doane']}")
    results.append(f"  → Consensus (median): {bins_dict_rect['consensus']} bins")
    
    results.append("\nGaussian Barrier - Bin Optimization:")
    results.append(f"  Sturges' rule:        {bins_dict_gauss['sturges']}")
    results.append(f"  Scott's rule:         {bins_dict_gauss['scott']}")
    results.append(f"  Freedman-Diaconis:    {bins_dict_gauss['freedman_diaconis']}")
    results.append(f"  Rice rule:            {bins_dict_gauss['rice']}")
    results.append(f"  Square root rule:     {bins_dict_gauss['sqrt']}")
    results.append(f"  Doane's formula:      {bins_dict_gauss['doane']}")
    results.append(f"  → Consensus (median): {bins_dict_gauss['consensus']} bins")
    
    # Shannon Entropy
    results.append("\n" + "-" * 80)
    results.append("1. SHANNON ENTROPY")
    results.append("-" * 80)
    results.append("   Measures the uncertainty/randomness in the probability distribution.")
    results.append("   H(X) = -∑ p(x) * log₂(p(x))")
    results.append("   Higher entropy = more uniform distribution")
    
    H_rect, bins_used_rect = shannon_entropy(prob_rect_filtered, bins='auto')
    H_gauss, bins_used_gauss = shannon_entropy(prob_gauss_filtered, bins='auto')
    
    results.append(f"\n   Rectangular Barrier:")
    results.append(f"     Shannon Entropy: {H_rect:.4f} bits")
    results.append(f"     Bins used:       {bins_used_rect}")
    
    results.append(f"\n   Gaussian Barrier:")
    results.append(f"     Shannon Entropy: {H_gauss:.4f} bits")
    results.append(f"     Bins used:       {bins_used_gauss}")
    
    results.append(f"\n   Entropy Difference: {abs(H_rect - H_gauss):.4f} bits")
    if H_rect > H_gauss:
        results.append("   → Rectangular barrier shows HIGHER entropy (more dispersed)")
    else:
        results.append("   → Gaussian barrier shows HIGHER entropy (more dispersed)")
    
    # Kullback-Leibler Divergence
    results.append("\n" + "-" * 80)
    results.append("2. KULLBACK-LEIBLER DIVERGENCE")
    results.append("-" * 80)
    results.append("   Measures how one distribution diverges from another (asymmetric).")
    results.append("   KL(P||Q) = ∑ p(x) * log₂(p(x)/q(x))")
    results.append("   Always ≥ 0; KL = 0 iff distributions are identical")
    
    kl_rect_gauss, bins_kl_rg = kl_divergence(prob_rect_filtered, prob_gauss_filtered, bins='auto')
    kl_gauss_rect, bins_kl_gr = kl_divergence(prob_gauss_filtered, prob_rect_filtered, bins='auto')
    
    results.append(f"\n   KL(Rectangular || Gaussian): {kl_rect_gauss:.4f} bits")
    results.append(f"   KL(Gaussian || Rectangular): {kl_gauss_rect:.4f} bits")
    results.append(f"   Bins used: {bins_kl_rg}")
    results.append(f"\n   Asymmetry: {abs(kl_rect_gauss - kl_gauss_rect):.4f} bits")
    
    if kl_rect_gauss > 0.1 or kl_gauss_rect > 0.1:
        results.append("   → Distributions show SIGNIFICANT divergence")
    else:
        results.append("   → Distributions show MINIMAL divergence")
    
    # Jensen-Shannon Divergence
    results.append("\n" + "-" * 80)
    results.append("3. JENSEN-SHANNON DIVERGENCE")
    results.append("-" * 80)
    results.append("   Symmetric version of KL divergence, bounded in [0, 1].")
    results.append("   JS(P||Q) = 0.5*KL(P||M) + 0.5*KL(Q||M), M = 0.5*(P+Q)")
    results.append("   JS = 0: identical; JS = 1: maximally different")
    
    js_div, bins_js = js_divergence(prob_rect_filtered, prob_gauss_filtered, bins='auto')
    js_distance = np.sqrt(js_div)  # JS distance (metric property)
    
    results.append(f"\n   JS Divergence: {js_div:.4f} bits")
    results.append(f"   JS Distance:   {js_distance:.4f} (metric version)")
    results.append(f"   Bins used:     {bins_js}")
    
    if js_div < 0.05:
        js_interp = "VERY SIMILAR distributions"
    elif js_div < 0.15:
        js_interp = "MODERATELY SIMILAR distributions"
    elif js_div < 0.30:
        js_interp = "MODERATELY DIFFERENT distributions"
    else:
        js_interp = "HIGHLY DIFFERENT distributions"
    
    results.append(f"   → {js_interp}")
    
    # Cross-Entropy
    results.append("\n" + "-" * 80)
    results.append("4. CROSS-ENTROPY")
    results.append("-" * 80)
    results.append("   Measures the average number of bits to encode samples from P using Q.")
    results.append("   H(P,Q) = -∑ p(x) * log₂(q(x))")
    
    ce_rect_gauss, bins_ce_rg = cross_entropy(prob_rect_filtered, prob_gauss_filtered, bins='auto')
    ce_gauss_rect, bins_ce_gr = cross_entropy(prob_gauss_filtered, prob_rect_filtered, bins='auto')
    
    results.append(f"\n   H(Rectangular, Gaussian): {ce_rect_gauss:.4f} bits")
    results.append(f"   H(Gaussian, Rectangular): {ce_gauss_rect:.4f} bits")
    results.append(f"   Bins used: {bins_ce_rg}")
    
    results.append(f"\n   Cross-entropy vs Shannon entropy:")
    results.append(f"   H(R,G) - H(R) = {ce_rect_gauss - H_rect:.4f} bits (info loss using G for R)")
    results.append(f"   H(G,R) - H(G) = {ce_gauss_rect - H_gauss:.4f} bits (info loss using R for G)")
    
    # Mutual Information (if meaningful for the data structure)
    # For flattened spatial-temporal data, this represents info about co-occurrence patterns
    results.append("\n" + "-" * 80)
    results.append("5. RELATIVE ENTROPY MEASURES")
    results.append("-" * 80)
    
    # Normalized entropy (compares to maximum possible entropy)
    max_entropy_rect = np.log2(bins_used_rect)
    max_entropy_gauss = np.log2(bins_used_gauss)
    norm_entropy_rect = H_rect / max_entropy_rect if max_entropy_rect > 0 else 0
    norm_entropy_gauss = H_gauss / max_entropy_gauss if max_entropy_gauss > 0 else 0
    
    results.append(f"\n   Normalized Shannon Entropy (H/H_max):")
    results.append(f"   Rectangular: {norm_entropy_rect:.4f} (max possible: {max_entropy_rect:.4f} bits)")
    results.append(f"   Gaussian:    {norm_entropy_gauss:.4f} (max possible: {max_entropy_gauss:.4f} bits)")
    results.append(f"\n   Interpretation:")
    results.append(f"   - Values close to 1.0 indicate nearly uniform distribution")
    results.append(f"   - Lower values indicate more structure/peaking in the distribution")
    
    # ============================================================
    # STANDARD STATISTICAL TESTS (Optimized with sampling)
    # ============================================================
    
    results.append("\n" + "=" * 80)
    results.append("HYPOTHESIS TESTING")
    results.append("=" * 80)
    
    alpha = 0.05
    results.append(f"\nSignificance level: α = {alpha}")
    
    # 1. Kolmogorov-Smirnov Test
    results.append("\n" + "-" * 80)
    results.append("1. KOLMOGOROV-SMIRNOV TEST (Two-Sample)")
    results.append("-" * 80)
    results.append("   Tests if two samples come from the same distribution.")
    results.append("   H0: Both samples from same continuous distribution")
    results.append("   H1: Samples from different distributions")
    
    ks_stat, ks_pval = stats.ks_2samp(prob_rect_filtered, prob_gauss_filtered)
    results.append(f"\n   KS Statistic: {ks_stat:.4f}")
    results.append(f"   P-value:      {ks_pval:.3e}")
    
    if ks_pval < alpha:
        results.append(f"   Decision:     REJECT H0 (p < {alpha})")
        results.append("   Interpretation: Distributions are SIGNIFICANTLY DIFFERENT.")
    else:
        results.append(f"   Decision:     FAIL TO REJECT H0 (p ≥ {alpha})")
        results.append("   Interpretation: No significant difference detected.")
    
    # 2. Mann-Whitney U Test
    results.append("\n" + "-" * 80)
    results.append("2. MANN-WHITNEY U TEST")
    results.append("-" * 80)
    results.append("   Non-parametric test for difference in central tendency.")
    results.append("   H0: Distributions have equal medians")
    results.append("   H1: Distributions have different medians")
    
    mw_stat, mw_pval = stats.mannwhitneyu(prob_rect_filtered, prob_gauss_filtered, alternative='two-sided')
    results.append(f"\n   U Statistic: {mw_stat:.3e}")
    results.append(f"   P-value:     {mw_pval:.3e}")
    
    if mw_pval < alpha:
        results.append(f"   Decision:     REJECT H0 (p < {alpha})")
        results.append("   Interpretation: Medians are SIGNIFICANTLY DIFFERENT.")
        if np.median(prob_rect_filtered) > np.median(prob_gauss_filtered):
            results.append("                   Rectangular barrier has HIGHER median probability.")
        else:
            results.append("                   Gaussian barrier has HIGHER median probability.")
    else:
        results.append(f"   Decision:     FAIL TO REJECT H0 (p ≥ {alpha})")
        results.append("   Interpretation: No significant difference in medians.")
    
    # 3. Kruskal-Wallis H Test
    results.append("\n" + "-" * 80)
    results.append("3. KRUSKAL-WALLIS H TEST")
    results.append("-" * 80)
    results.append("   Extension of Mann-Whitney to multiple groups.")
    results.append("   H0: All groups have the same median")
    results.append("   H1: At least one group differs")
    
    kw_stat, kw_pval = stats.kruskal(prob_rect_filtered, prob_gauss_filtered)
    results.append(f"\n   H Statistic: {kw_stat:.4f}")
    results.append(f"   P-value:     {kw_pval:.3e}")
    
    if kw_pval < alpha:
        results.append(f"   Decision:     REJECT H0 (p < {alpha})")
        results.append("   Interpretation: Groups are SIGNIFICANTLY DIFFERENT.")
    else:
        results.append(f"   Decision:     FAIL TO REJECT H0 (p ≥ {alpha})")
        results.append("   Interpretation: No significant group difference.")
    
    # 4. Mood's Test (for equal scale)
    results.append("\n" + "-" * 80)
    results.append("4. MOOD'S TEST")
    results.append("-" * 80)
    results.append("   Tests for equal scale parameters (spread/variance).")
    results.append("   H0: Samples have equal scale")
    results.append("   H1: Samples have different scale")
    
    mood_stat, mood_pval = stats.mood(prob_rect_filtered, prob_gauss_filtered)
    results.append(f"\n   Mood Statistic: {mood_stat:.4f}")
    results.append(f"   P-value:        {mood_pval:.3e}")
    
    if mood_pval < alpha:
        results.append(f"   Decision:     REJECT H0 (p < {alpha})")
        results.append("   Interpretation: Scales are SIGNIFICANTLY DIFFERENT.")
    else:
        results.append(f"   Decision:     FAIL TO REJECT H0 (p ≥ {alpha})")
        results.append("   Interpretation: No significant difference in scale.")
    
    # 5. Anderson-Darling Test (two-sample)
    results.append("\n" + "-" * 80)
    results.append("5. ANDERSON-DARLING TEST")
    results.append("-" * 80)
    results.append("   Tests if samples come from the same distribution.")
    results.append("   More sensitive to tails than KS test.")
    results.append("   H0: Samples from same distribution")
    results.append("   H1: Samples from different distributions")
    
    ad_result = stats.anderson_ksamp([prob_rect_filtered, prob_gauss_filtered])
    results.append(f"\n   AD Statistic:     {ad_result.statistic:.4f}")
    results.append(f"   Critical values:  {ad_result.critical_values}")
    results.append(f"   Significance:     {ad_result.significance_level}")
    
    if ad_result.statistic > ad_result.critical_values[2]:  # 5% level
        results.append(f"   Decision:     REJECT H0 (statistic > critical value at 5%)")
        results.append("   Interpretation: Distributions are SIGNIFICANTLY DIFFERENT.")
    else:
        results.append(f"   Decision:     FAIL TO REJECT H0")
        results.append("   Interpretation: No significant difference detected.")
    
    # 6. Epps-Singleton Test
    results.append("\n" + "-" * 80)
    results.append("6. EPPS-SINGLETON TEST")
    results.append("-" * 80)
    results.append("   Tests for equal distributions based on characteristic functions.")
    results.append("   H0: Samples have same distribution")
    results.append("   H1: Distributions differ")
    
    es_stat, es_pval = stats.epps_singleton_2samp(prob_rect_filtered, prob_gauss_filtered)
    results.append(f"\n   ES Statistic: {es_stat:.4f}")
    results.append(f"   P-value:      {es_pval:.3e}")
    
    if es_pval < alpha:
        results.append(f"   Decision:     REJECT H0 (p < {alpha})")
        results.append("   Interpretation: Distributions are SIGNIFICANTLY DIFFERENT.")
    else:
        results.append(f"   Decision:     FAIL TO REJECT H0 (p ≥ {alpha})")
        results.append("   Interpretation: No significant difference detected.")
    
    # 7. Ansari-Bradley Test
    results.append("\n" + "-" * 80)
    results.append("7. ANSARI-BRADLEY TEST")
    results.append("-" * 80)
    results.append("   Tests if two samples have same scale (dispersion).")
    results.append("   H0: Samples have equal scale parameters")
    results.append("   H1: Scale parameters differ")
    
    ab_stat, ab_pval = stats.ansari(prob_rect_filtered, prob_gauss_filtered)
    results.append(f"\n   AB Statistic: {ab_stat:.3f}")
    results.append(f"   P-value:      {ab_pval:.3e}")
    
    if ab_pval < alpha:
        results.append(f"   Decision:     REJECT H0 (p < {alpha})")
        results.append("   Interpretation: Dispersions are SIGNIFICANTLY DIFFERENT.")
        if np.std(prob_rect_filtered) > np.std(prob_gauss_filtered):
            results.append("                   Rectangular barrier shows GREATER variability.")
        else:
            results.append("                   Gaussian barrier shows GREATER variability.")
    else:
        results.append(f"   Decision:     FAIL TO REJECT H0 (p ≥ {alpha})")
        results.append("   Interpretation: No significant difference in dispersion.")
    
    # EFFECT SIZE MEASURES
    results.append("\n" + "=" * 80)
    results.append("EFFECT SIZE MEASURES")
    results.append("=" * 80)
    
    # Cliff's Delta
    def cliffs_delta(x, y, sample_size=10000):
        """Calculate Cliff's Delta with sampling for large datasets"""
        if len(x) > sample_size:
            x = np.random.choice(x, sample_size, replace=False)
        if len(y) > sample_size:
            y = np.random.choice(y, sample_size, replace=False)
        
        n_x, n_y = len(x), len(y)
        dominance = sum(1 for xi in x for yi in y if xi > yi) - sum(1 for xi in x for yi in y if xi < yi)
        return dominance / (n_x * n_y)
    
    cliff_delta = cliffs_delta(prob_rect_filtered, prob_gauss_filtered)
    results.append(f"\nCliff's Delta: {cliff_delta:.4f}")
    results.append("   Range: [-1, 1]")
    results.append("   Interpretation guidelines:")
    results.append("   - |δ| < 0.147: NEGLIGIBLE effect")
    results.append("   - 0.147 ≤ |δ| < 0.330: SMALL effect")
    results.append("   - 0.330 ≤ |δ| < 0.474: MEDIUM effect")
    results.append("   - |δ| ≥ 0.474: LARGE effect")
    
    if abs(cliff_delta) < 0.147:
        cliff_interp = "NEGLIGIBLE effect size"
    elif abs(cliff_delta) < 0.330:
        cliff_interp = "SMALL effect size"
    elif abs(cliff_delta) < 0.474:
        cliff_interp = "MEDIUM effect size"
    else:
        cliff_interp = "LARGE effect size"
    results.append(f"\n   Result: {cliff_interp}")
    
    if cliff_delta > 0:
        results.append("   Direction: Rectangular barrier has HIGHER probability values")
    elif cliff_delta < 0:
        results.append("   Direction: Gaussian barrier has HIGHER probability values")
    else:
        results.append("   Direction: No dominance detected")
    
    # SUMMARY AND CONCLUSIONS
    results.append("\n" + "=" * 80)
    results.append("SUMMARY AND CONCLUSIONS")
    results.append("=" * 80)
    
    results.append("\n--- Key Findings ---")
    results.append(f"1. Mean probability density:")
    results.append(f"   - Rectangular: {np.mean(prob_rect_filtered):.6e} nm⁻¹")
    results.append(f"   - Gaussian:    {np.mean(prob_gauss_filtered):.6e} nm⁻¹")
    results.append(f"   - Difference:  {abs(np.mean(prob_rect_filtered) - np.mean(prob_gauss_filtered)):.6e} nm⁻¹")
    
    results.append(f"\n2. Median probability density:")
    results.append(f"   - Rectangular: {np.median(prob_rect_filtered):.6e} nm⁻¹")
    results.append(f"   - Gaussian:    {np.median(prob_gauss_filtered):.6e} nm⁻¹")
    results.append(f"   - Difference:  {abs(np.median(prob_rect_filtered) - np.median(prob_gauss_filtered)):.6e} nm⁻¹")
    
    results.append(f"\n3. Variability (Std Dev):")
    results.append(f"   - Rectangular: {np.std(prob_rect_filtered):.6e} nm⁻¹")
    results.append(f"   - Gaussian:    {np.std(prob_gauss_filtered):.6e} nm⁻¹")
    
    results.append(f"\n4. Information-theoretic measures:")
    results.append(f"   - Shannon Entropy difference:  {abs(H_rect - H_gauss):.4f} bits")
    results.append(f"   - Jensen-Shannon Divergence:   {js_div:.4f} bits")
    results.append(f"   - KL Divergence (R||G):        {kl_rect_gauss:.4f} bits")
    results.append(f"   - KL Divergence (G||R):        {kl_gauss_rect:.4f} bits")
    
    sig_tests = sum([
        ks_pval < alpha,
        mw_pval < alpha,
        kw_pval < alpha,
        mood_pval < alpha,
        es_pval < alpha,
        ab_pval < alpha,
        ad_result.statistic > ad_result.critical_values[2]
    ])
    
    results.append(f"\n5. Statistical significance:")
    results.append(f"   - {sig_tests}/7 tests showed significant differences (α = {alpha})")
    
    if sig_tests >= 5:
        results.append("\n--- OVERALL CONCLUSION ---")
        results.append("The probability distributions from rectangular and Gaussian barriers are")
        results.append("SIGNIFICANTLY DIFFERENT. The barrier geometry has a SUBSTANTIAL impact on")
        results.append("the quantum tunneling probability distribution.")
    elif sig_tests >= 3:
        results.append("\n--- OVERALL CONCLUSION ---")
        results.append("There is MODERATE to STRONG EVIDENCE that probability distributions differ")
        results.append("between rectangular and Gaussian barriers. The barrier geometry appears to")
        results.append("influence the tunneling characteristics notably.")
    else:
        results.append("\n--- OVERALL CONCLUSION ---")
        results.append("The probability distributions show LIMITED EVIDENCE of difference between")
        results.append("rectangular and Gaussian barriers. The barrier geometry may have modest")
        results.append("impact on the overall tunneling probability distribution.")
    
    results.append("\n--- Physical Interpretation ---")
    results.append("The quantum tunneling process is sensitive to the detailed shape of the")
    results.append("potential barrier. Rectangular barriers have sharp edges that can lead to")
    results.append("different interference patterns compared to the smooth Gaussian profile.")
    results.append("These geometric differences manifest in the spatial and temporal evolution")
    results.append("of the probability density, as quantified by both classical statistical")
    results.append("measures and information-theoretic entropy metrics.")
    
    results.append("\n--- Entropy-Based Insights ---")
    if abs(H_rect - H_gauss) > 0.5:
        results.append("The substantial entropy difference suggests the barrier shapes produce")
        results.append("markedly different degrees of spatial localization in the wavefunction.")
    else:
        results.append("The similar entropy values suggest comparable degrees of wavefunction")
        results.append("spreading, despite different barrier geometries.")
    
    if js_div > 0.2:
        results.append("The high Jensen-Shannon divergence confirms these are distinct quantum")
        results.append("mechanical regimes with different tunneling dynamics.")
    else:
        results.append("The low Jensen-Shannon divergence suggests similar overall tunneling")
        results.append("behavior despite geometric differences.")
    
    # Add metadata from NetCDF
    results.append("\n" + "=" * 80)
    results.append("SIMULATION METADATA")
    results.append("=" * 80)
    
    results.append("\n--- Rectangular Barrier ---")
    results.append(f"Transmission coefficient: {float(rec_barrier.transmission):.6f}")
    results.append(f"Reflection coefficient:   {float(rec_barrier.reflection):.6f}")
    results.append(f"Absorbed probability:     {float(rec_barrier.absorbed):.6f}")
    results.append(f"Barrier type:             {rec_barrier.barrier_type}")
    results.append(f"Scenario:                 {rec_barrier.scenario}")
    
    results.append("\n--- Gaussian Barrier ---")
    results.append(f"Transmission coefficient: {float(gauss_barrier.transmission):.6f}")
    results.append(f"Reflection coefficient:   {float(gauss_barrier.reflection):.6f}")
    results.append(f"Absorbed probability:     {float(gauss_barrier.absorbed):.6f}")
    results.append(f"Barrier type:             {gauss_barrier.barrier_type}")
    results.append(f"Scenario:                 {gauss_barrier.scenario}")
    
    results.append("\n" + "=" * 80)
    results.append("END OF STATISTICAL ANALYSIS")
    results.append("=" * 80)
    
    # Write results to file
    print("\n[6/6] Saving statistical analysis...")
    output_file = "../stats/comparison_stats.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results))
    
    print(f"✓ Statistical analysis saved: {output_file}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Figure saved:     ../figs/probability_comparison.png (600 DPI)")
    print(f"Statistics saved: ../stats/comparison_stats.txt")
    print(f"")
    print(f"Key Results:")
    print(f"  • {sig_tests}/7 hypothesis tests significant (α={alpha})")
    print(f"  • Shannon entropy diff: {abs(H_rect - H_gauss):.4f} bits")
    print(f"  • JS divergence: {js_div:.4f} bits")
    print(f"  • Effect size (Cliff's δ): {cliff_delta:.4f} ({cliff_interp})")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
