#!/usr/bin/env python
"""
Publication-Quality Quantum Tunneling Visualization
Generates 2x2 subplot figures for rectangular and Gaussian barrier cases
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import netCDF4 as nc

# Publication-quality settings
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.linewidth'] = 1.2
mpl.rcParams['xtick.major.width'] = 1.2
mpl.rcParams['ytick.major.width'] = 1.2
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['mathtext.fontset'] = 'stix'


def find_key_frames(x, t, probability, potential):
    """
    Identify 4 key frames for visualization:
    (a) Beginning
    (b) Right before hitting barrier
    (c) Right after hitting barrier (peak interaction)
    (d) End
    """
    n_frames = len(t)
    
    # Frame (a): Beginning
    frame_a = 0
    
    # Find barrier center
    V_max = np.max(potential)
    barrier_mask = potential > 0.1 * V_max
    if np.any(barrier_mask):
        barrier_indices = np.where(barrier_mask)[0]
        barrier_center_idx = (barrier_indices[0] + barrier_indices[-1]) // 2
    else:
        barrier_center_idx = len(x) // 2
    
    # Find frames based on probability center of mass
    prob_com = np.zeros(n_frames)
    for i in range(n_frames):
        prob_com[i] = np.sum(x * probability[i]) / (np.sum(probability[i]) + 1e-12)
    
    x_barrier_center = x[barrier_center_idx]
    
    # Frame (b): Right before hitting barrier (COM approaches barrier)
    # Find when COM is ~2-3 nm before barrier center
    before_barrier_frames = np.where(prob_com < x_barrier_center - 1.5)[0]
    if len(before_barrier_frames) > 0:
        frame_b = before_barrier_frames[-1]
    else:
        frame_b = n_frames // 4
    
    # Frame (c): Peak interaction (maximum probability at barrier)
    prob_at_barrier = probability[:, barrier_center_idx]
    frame_c = np.argmax(prob_at_barrier)
    
    # Frame (d): End
    frame_d = n_frames - 1
    
    return frame_a, frame_b, frame_c, frame_d


def compute_statistics(x, t, probability, potential, psi_real, psi_imag, frame_idx):
    """
    Compute physical statistics at given frame.
    """
    dx = x[1] - x[0]
    prob = probability[frame_idx]
    psi = psi_real[frame_idx] + 1j * psi_imag[frame_idx]
    
    # Normalization
    norm = np.sum(prob) * dx
    
    # Position expectation and variance
    x_mean = np.sum(x * prob) * dx
    x2_mean = np.sum(x**2 * prob) * dx
    x_std = np.sqrt(np.abs(x2_mean - x_mean**2))
    
    # Momentum expectation (from derivative)
    psi_gradient = np.gradient(psi, dx)
    k_mean = np.sum(prob * np.imag(np.conj(psi) * psi_gradient)) * dx
    
    # Kinetic energy
    E_kinetic = 0.5 * np.sum(np.abs(psi_gradient)**2) * dx
    
    # Potential energy
    E_potential = np.sum(potential * prob) * dx
    
    # Total energy
    E_total = E_kinetic + E_potential
    
    # Maximum probability density
    prob_max = np.max(prob)
    x_max = x[np.argmax(prob)]
    
    stats = {
        'time': t[frame_idx],
        'norm': norm,
        'x_mean': x_mean,
        'x_std': x_std,
        'k_mean': k_mean,
        'E_kinetic': E_kinetic,
        'E_potential': E_potential,
        'E_total': E_total,
        'prob_max': prob_max,
        'x_max': x_max
    }
    
    return stats


def get_global_ylim(nc_files):
    """
    Determine global y-axis limits across all experiments.
    """
    global_prob_max = 0.0
    
    for nc_file in nc_files:
        if nc_file.exists():
            data = nc.Dataset(nc_file, 'r')
            probability = data['probability'][:]
            prob_max = np.max(probability)
            global_prob_max = max(global_prob_max, prob_max)
            data.close()
    
    return 0, global_prob_max * 1.1


def create_publication_figure(nc_file, output_name, figs_dir, ylim):
    """
    Create 2x2 publication-quality figure.
    Returns statistics dictionary for combined report.
    """
    # Read NetCDF data
    data = nc.Dataset(nc_file, 'r')
    x = data['x'][:]
    t = data['t'][:]
    probability = data['probability'][:]
    potential = data['potential'][:]
    psi_real = data['psi_real'][:]
    psi_imag = data['psi_imag'][:]
    
    T = data.transmission
    R = data.reflection
    A = data.absorbed
    
    scenario_name = data.scenario
    barrier_type = data.barrier_type
    
    data.close()
    
    # Find key frames
    frame_a, frame_b, frame_c, frame_d = find_key_frames(
        x, t, probability, potential
    )
    frames = [frame_a, frame_b, frame_c, frame_d]
    labels = ['(a)', '(b)', '(c)', '(d)']
    
    # Setup figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), facecolor='white')
    axes = axes.flatten()
    
    # Color scheme for publication
    prob_color = '#2E86AB'      # Professional blue
    barrier_color = '#A23B72'   # Deep magenta
    fill_color = '#87BCDE'      # Light blue
    
    # Scaling for potential overlay
    V_max = np.max(potential)
    V_scale = 0.35 * ylim[1] / (V_max + 1e-10)
    
    # Statistics storage
    all_stats = []
    
    # Plot each frame
    for idx, (frame, label, ax) in enumerate(zip(frames, labels, axes)):
        prob = probability[frame]
        t_val = t[frame]
        
        # Plot probability density
        ax.fill_between(x, 0, prob, color=fill_color, alpha=0.4, 
                       linewidth=0, zorder=2)
        ax.plot(x, prob, color=prob_color, linewidth=2.0, 
               zorder=3, label=r'$|\psi(x,t)|^2$')
        
        # Plot potential barrier
        ax.fill_between(x, 0, potential * V_scale, 
                       color=barrier_color, alpha=0.25, 
                       linewidth=0, zorder=1)
        ax.plot(x, potential * V_scale, color=barrier_color, 
               linewidth=2.0, linestyle='-', zorder=1, 
               label=r'$V(x)$ (scaled)')
        
        # Formatting
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(ylim[0], ylim[1])
        
        ax.set_xlabel(r'Position $x$ [nm]', fontsize=11, fontweight='bold')
        ax.set_ylabel(r'$|\psi|^2$ [nm$^{-1}$]', fontsize=11, fontweight='bold')
        
        # Add subplot label (only this - no time annotation)
        ax.text(0.03, 0.97, label, transform=ax.transAxes,
               fontsize=14, fontweight='bold', va='top', ha='left',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                        edgecolor='black', linewidth=1.2))
        
        # Grid
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5, color='gray')
        
        # Legend only on first subplot
        if idx == 0:
            ax.legend(loc='upper right', fontsize=9, framealpha=0.95,
                     edgecolor='black', fancybox=False)
        
        # Compute statistics
        stats = compute_statistics(x, t, probability, potential,
                                   psi_real, psi_imag, frame)
        all_stats.append(stats)
    
    # Tight layout
    plt.tight_layout(pad=1.5)
    
    # Save figure
    fig_path = figs_dir / f'{output_name}.png'
    plt.savefig(fig_path, dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig)
    
    print(f"  ✓ Saved: {fig_path}")
    
    # Return stats for combined report
    return {
        'scenario_name': scenario_name,
        'barrier_type': barrier_type,
        'T': T,
        'R': R,
        'A': A,
        'frames': all_stats,
        'labels': labels
    }


def write_combined_statistics(all_case_stats, stats_dir):
    """
    Write combined statistics report for all cases.
    """
    stats_path = stats_dir / 'quantum_tunneling_statistics.txt'
    
    with open(stats_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("QUANTUM TUNNELING SIMULATION STATISTICS\n")
        f.write("Publication-Quality Analysis Report\n")
        f.write("="*70 + "\n\n")
        
        for case_idx, case_stats in enumerate(all_case_stats, 1):
            f.write(f"\n{'='*70}\n")
            f.write(f"CASE {case_idx}: {case_stats['scenario_name']}\n")
            f.write(f"Barrier Type: {case_stats['barrier_type']}\n")
            f.write(f"{'='*70}\n\n")
            
            f.write("Global Results:\n")
            f.write(f"  Transmission coefficient (T): {case_stats['T']:.3f}\n")
            f.write(f"  Reflection coefficient (R):   {case_stats['R']:.3f}\n")
            f.write(f"  Absorbed probability (A):     {case_stats['A']:.3f}\n")
            f.write(f"  Total (T+R+A):                {case_stats['T']+case_stats['R']+case_stats['A']:.3f}\n")
            f.write(f"\n{'-'*70}\n\n")
            
            for label, stats in zip(case_stats['labels'], case_stats['frames']):
                f.write(f"Frame {label}:\n")
                f.write(f"  Time:                    {stats['time']:.3f} fs\n")
                f.write(f"  Normalization:           {stats['norm']:.3f}\n")
                f.write(f"  Position <x>:            {stats['x_mean']:.3f} nm\n")
                f.write(f"  Position std σ_x:        {stats['x_std']:.3f} nm\n")
                f.write(f"  Momentum <k>:            {stats['k_mean']:.3f} nm^-1\n")
                f.write(f"  Kinetic energy:          {stats['E_kinetic']:.3f} eV\n")
                f.write(f"  Potential energy:        {stats['E_potential']:.3f} eV\n")
                f.write(f"  Total energy:            {stats['E_total']:.3f} eV\n")
                f.write(f"  Max probability density: {stats['prob_max']:.3f} nm^-1\n")
                f.write(f"  Position of max prob:    {stats['x_max']:.3f} nm\n")
                f.write(f"\n")
            
            f.write(f"{'-'*70}\n")
        
        f.write(f"\n{'='*70}\n")
        f.write("END OF REPORT\n")
        f.write(f"{'='*70}\n")
    
    print(f"  ✓ Saved combined statistics: {stats_path}")


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("PUBLICATION-QUALITY QUANTUM TUNNELING VISUALIZATION")
    print("="*70 + "\n")
    
    # Setup directories
    raw_data_dir = Path('../raw_data')
    figs_dir = Path('../figs')
    stats_dir = Path('../stats')
    
    figs_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    # Define cases
    cases = [
        ('case1_rectangular_barrier.nc', 'case1_rectangular_barrier'),
        ('case2_gaussian_barrier.nc', 'case2_gaussian_barrier')
    ]
    
    # Get file paths
    nc_files = [raw_data_dir / nc_filename for nc_filename, _ in cases]
    
    # Determine global y-axis limits
    print("Analyzing data for consistent y-axis scaling...")
    ylim = get_global_ylim(nc_files)
    print(f"  Global y-axis range: [{ylim[0]:.3f}, {ylim[1]:.3f}] nm^-1\n")
    
    # Process cases and collect statistics
    all_case_stats = []
    
    for nc_filename, output_name in cases:
        nc_path = raw_data_dir / nc_filename
        
        if not nc_path.exists():
            print(f"  ✗ File not found: {nc_path}")
            continue
        
        print(f"Processing: {nc_filename}")
        case_stats = create_publication_figure(nc_path, output_name, figs_dir, ylim)
        all_case_stats.append(case_stats)
        print()
    
    # Write combined statistics report
    print("Generating combined statistics report...")
    write_combined_statistics(all_case_stats, stats_dir)
    print()
    
    print("="*70)
    print("ALL FIGURES AND STATISTICS GENERATED SUCCESSFULLY")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
