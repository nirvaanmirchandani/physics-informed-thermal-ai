"""
Quick sanity-check plots for synthetic thermal dataset.

Minimal visualization of generated data to verify:
- Temperature continuity
- Power/cooling/temperature relationships
- Dataset validity
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.generate_dataset import ThermalDatasetGenerator


def plot_scenario_timeseries(df: pd.DataFrame, num_scenarios: int = 5, output_dir: str = 'data/figures'):
    """
    Plot temperature, power, and cooling intensity over time for multiple scenarios.
    
    Args:
        df (pd.DataFrame): Dataset
        num_scenarios (int): Number of scenarios to plot (default 5)
        output_dir (str): Directory to save figures
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_scenarios_available = df['scenario_id'].nunique()
    num_scenarios = min(num_scenarios, n_scenarios_available)
    scenario_ids = sorted(df['scenario_id'].unique())[:num_scenarios]
    
    # Temperature vs time
    fig, ax = plt.subplots(figsize=(12, 6))
    for sid in scenario_ids:
        scenario_data = df[df['scenario_id'] == sid]
        ax.plot(scenario_data['time'], scenario_data['temperature'], 
                label=f'Scenario {sid}', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Temperature (K)')
    ax.set_title(f'Temperature vs Time ({num_scenarios} scenarios)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'temperature_vs_time.png', dpi=100)
    plt.close(fig)
    print(f"  Saved: temperature_vs_time.png")
    
    # Power vs time
    fig, ax = plt.subplots(figsize=(12, 6))
    for sid in scenario_ids:
        scenario_data = df[df['scenario_id'] == sid]
        ax.plot(scenario_data['time'], scenario_data['power'], 
                label=f'Scenario {sid}', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Power (W)')
    ax.set_title(f'Power vs Time ({num_scenarios} scenarios)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'power_vs_time.png', dpi=100)
    plt.close(fig)
    print(f"  Saved: power_vs_time.png")
    
    # Cooling intensity vs time
    fig, ax = plt.subplots(figsize=(12, 6))
    for sid in scenario_ids:
        scenario_data = df[df['scenario_id'] == sid]
        ax.plot(scenario_data['time'], scenario_data['cooling_intensity'], 
                label=f'Scenario {sid}', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cooling Intensity [0, 1]')
    ax.set_title(f'Cooling Intensity vs Time ({num_scenarios} scenarios)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'cooling_vs_time.png', dpi=100)
    plt.close(fig)
    print(f"  Saved: cooling_vs_time.png")


def plot_temperature_vs_power(df: pd.DataFrame, output_dir: str = 'data/figures'):
    """
    Scatter plot of temperature vs power across all data.
    
    Args:
        df (pd.DataFrame): Dataset
        output_dir (str): Directory to save figures
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color by scenario for visibility
    unique_scenarios = df['scenario_id'].nunique()
    colors = plt.cm.tab20(np.linspace(0, 1, min(unique_scenarios, 20)))
    
    for i, sid in enumerate(sorted(df['scenario_id'].unique())[:20]):
        scenario_data = df[df['scenario_id'] == sid]
        ax.scatter(scenario_data['power'], scenario_data['temperature'], 
                  alpha=0.4, s=10, color=colors[i % len(colors)], 
                  label=f'Scenario {sid}' if i < 5 else '')
    
    ax.set_xlabel('Power (W)')
    ax.set_ylabel('Temperature (K)')
    ax.set_title('Temperature vs Power (all scenarios)')
    if len(df['scenario_id'].unique()) <= 5:
        ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'temperature_vs_power.png', dpi=100)
    plt.close(fig)
    print(f"  Saved: temperature_vs_power.png")


def print_dataset_summary(df: pd.DataFrame):
    """Print basic summary statistics."""
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    
    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Scenarios: {df['scenario_id'].nunique()}")
    print(f"Timesteps per scenario: {len(df[df['scenario_id'] == 0])}")
    
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  ✓ None (dataset complete)")
    else:
        print(f"  {missing[missing > 0].to_dict()}")
    
    print(f"\nTemperature (K):")
    print(f"  Min: {df['temperature'].min():.2f} ({df['temperature'].min() - 273.15:.2f}°C)")
    print(f"  Max: {df['temperature'].max():.2f} ({df['temperature'].max() - 273.15:.2f}°C)")
    print(f"  Mean: {df['temperature'].mean():.2f} ({df['temperature'].mean() - 273.15:.2f}°C)")
    print(f"  Std: {df['temperature'].std():.2f} K")
    
    print(f"\nPower (W):")
    print(f"  Min: {df['power'].min():.1f}")
    print(f"  Max: {df['power'].max():.1f}")
    print(f"  Mean: {df['power'].mean():.1f}")
    print(f"  Std: {df['power'].std():.1f}")
    
    print(f"\nCooling Intensity [0, 1]:")
    print(f"  Min: {df['cooling_intensity'].min():.4f}")
    print(f"  Max: {df['cooling_intensity'].max():.4f}")
    print(f"  Mean: {df['cooling_intensity'].mean():.4f}")
    print(f"  Std: {df['cooling_intensity'].std():.4f}")
    
    print(f"\nSystem Parameters (per scenario):")
    print(f"  C (J/K): {df['C'].min():.0f}–{df['C'].max():.0f} (mean {df['C'].mean():.0f})")
    print(f"  hA (W/K): {df['hA'].min():.1f}–{df['hA'].max():.1f} (mean {df['hA'].mean():.1f})")
    print(f"  T_ambient (K): {df['T_ambient'].min():.2f}–{df['T_ambient'].max():.2f}")


def check_temperature_continuity(df: pd.DataFrame):
    """Check that temperature evolves smoothly within each scenario."""
    print("\n" + "="*60)
    print("TEMPERATURE CONTINUITY CHECK")
    print("="*60)
    
    max_jump = 0
    scenario_with_max_jump = None
    
    for scenario_id in df['scenario_id'].unique():
        scenario_data = df[df['scenario_id'] == scenario_id].sort_values('timestep')
        temps = scenario_data['temperature'].values
        
        if len(temps) > 1:
            jumps = np.abs(np.diff(temps))
            scenario_max_jump = np.max(jumps)
            
            if scenario_max_jump > max_jump:
                max_jump = scenario_max_jump
                scenario_with_max_jump = scenario_id
    
    print(f"\nMax temperature jump in any scenario: {max_jump:.6f} K")
    print(f"  (Scenario {scenario_with_max_jump})")
    
    if max_jump < 0.1:
        print("  ✓ Excellent: Temperature is smooth and continuous")
    elif max_jump < 0.5:
        print("  ✓ Good: Temperature evolution is reasonably smooth")
    else:
        print(f"  ⚠ Warning: Large temperature jumps detected")


def check_power_cooling_effect(df: pd.DataFrame):
    """Check that power and cooling have expected effects on temperature."""
    print("\n" + "="*60)
    print("POWER & COOLING EFFECT CHECK")
    print("="*60)
    
    # Correlation checks
    corr_power_temp = df['power'].corr(df['temperature'])
    corr_cooling_temp = df['cooling_intensity'].corr(df['temperature'])
    
    print(f"\nCorrelation (power, temperature): {corr_power_temp:.4f}")
    if corr_power_temp > 0.3:
        print("  ✓ Strong positive: Higher power → Higher temperature")
    elif corr_power_temp > 0.1:
        print("  ✓ Moderate positive: Power affects temperature")
    else:
        print("  ⚠ Weak: Expected stronger power-temp relationship")
    
    print(f"\nCorrelation (cooling intensity, temperature): {corr_cooling_temp:.4f}")
    if corr_cooling_temp < -0.2:
        print("  ✓ Strong negative: Higher cooling → Lower temperature")
    elif corr_cooling_temp < -0.05:
        print("  ✓ Moderate negative: Cooling affects temperature")
    else:
        print("  ⚠ Weak: Expected stronger cooling-temp relationship")


def main():
    """Generate dataset and create sanity-check plots."""
    print("\n" + "="*60)
    print("STAGE 2B SANITY CHECK: SYNTHETIC DATASET")
    print("="*60)
    
    # Generate dataset
    print("\nGenerating synthetic dataset...")
    print("  (100 scenarios, 1000 steps each, seed=42)")
    
    gen = ThermalDatasetGenerator(seed=42)
    df = gen.generate(n_scenarios=100, n_steps_per_scenario=1000, dt=1.0)
    
    # Print summary
    print_dataset_summary(df)
    
    # Check continuity
    check_temperature_continuity(df)
    
    # Check power/cooling effects
    check_power_cooling_effect(df)
    
    # Generate plots
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60 + "\n")
    
    plot_scenario_timeseries(df, num_scenarios=5)
    plot_temperature_vs_power(df)
    
    # Final notes
    print("\n" + "="*60)
    print("KNOWN LIMITATIONS")
    print("="*60)
    print("""
  1. "Reactive" cooling pattern is randomized, not truly reactive.
     (A true reactive pattern would adjust cooling based on measured temperature.)
  
  2. Data is physics-consistent but NOT empirically validated.
     (Not based on real datacenter measurements.)
  
  3. All scenarios are independent (no long-term drift or memory effects).
  
  4. Cooling boost (10 K) is hardcoded (not empirically derived).
    """)
    
    print("\n" + "="*60)
    print("SANITY CHECK COMPLETE")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
