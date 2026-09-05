"""
Synthetic thermal dataset generator.

Generates physics-consistent thermal scenarios using the Stage 1 ThermalModel.
Each scenario has:
- Randomly varied system parameters (C, hA, T_ambient)
- Varied workload patterns (steps, ramps, spikes, periodic)
- Varied cooling strategies (constant, reactive, ramps)
- Full time-series trajectory with all variables

The resulting dataset is suitable for training ML models to predict temperature
given power and cooling inputs.

All randomness is seeded and reproducible.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator import ThermalModel


class ThermalDatasetGenerator:
    """
    Generate physics-consistent synthetic thermal datasets.
    
    The generator creates multiple independent scenarios by:
    1. Randomly sampling thermal system parameters (C, hA, T_ambient)
    2. Creating time-varying workload patterns (power sequences)
    3. Creating time-varying cooling strategies (cooling_intensity sequences)
    4. Running the ThermalModel simulator for each scenario
    5. Collecting the resulting time-series into a DataFrame
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the dataset generator.
        
        Args:
            seed (int): Random seed for reproducibility. Default 42.
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.scenarios = []
        self.data_rows = []
    
    def _sample_system_parameters(self) -> Tuple[float, float, float, float]:
        """
        Sample random but physically reasonable system parameters.
        
        Returns:
            Tuple of (C, hA, T_ambient, T_initial)
        """
        # Thermal capacitance: 1000–10000 J/K
        C = self.rng.uniform(1000, 10000)
        
        # Heat removal coefficient: 50–200 W/K
        hA = self.rng.uniform(50, 200)
        
        # Ambient temperature: 288–303 K (15–30°C, realistic datacenter range)
        T_ambient = self.rng.uniform(288.15, 303.15)
        
        # Initial temperature: within ±5 K of ambient, but >= 273.15 K
        T_initial = np.clip(T_ambient + self.rng.uniform(-5, 5), 273.15, 370.0)
        
        return C, hA, T_ambient, T_initial
    
    def _generate_power_sequence(self, n_steps: int, scenario_type: str) -> np.ndarray:
        """
        Generate time-varying server power sequence.
        
        Args:
            n_steps (int): Number of timesteps.
            scenario_type (str): Type of power pattern:
                - 'low_constant': constant low power
                - 'normal_constant': constant medium power
                - 'high_constant': constant high power
                - 'step': sudden power change
                - 'spike': brief power spike
                - 'ramp': gradual power increase
                - 'periodic': sinusoidal power variation
        
        Returns:
            np.ndarray: Power sequence (W), shape (n_steps,)
        """
        power = np.zeros(n_steps)
        
        if scenario_type == 'low_constant':
            power = np.full(n_steps, 500.0)  # 500 W
        
        elif scenario_type == 'normal_constant':
            power = np.full(n_steps, 1200.0)  # 1.2 kW
        
        elif scenario_type == 'high_constant':
            power = np.full(n_steps, 2000.0)  # 2 kW
        
        elif scenario_type == 'step':
            # Low power for first half, high for second half
            mid = n_steps // 2
            power[:mid] = 800.0
            power[mid:] = 1800.0
        
        elif scenario_type == 'spike':
            # Normal power with brief spike in middle
            power = np.full(n_steps, 1000.0)
            spike_start = n_steps // 3
            spike_end = 2 * n_steps // 3
            power[spike_start:spike_end] = 2200.0
        
        elif scenario_type == 'ramp':
            # Gradual ramp from low to high
            power = np.linspace(600.0, 1900.0, n_steps)
        
        elif scenario_type == 'periodic':
            # Sinusoidal variation around mean
            mean_power = 1200.0
            amplitude = 500.0
            t = np.linspace(0, 4 * np.pi, n_steps)  # 2 full periods
            power = mean_power + amplitude * np.sin(t)
            power = np.clip(power, 100.0, 2500.0)  # Clamp to valid range
        
        else:
            raise ValueError(f"Unknown scenario_type: {scenario_type}")
        
        # Add small random noise (~5% std)
        noise = self.rng.normal(0, 0.05 * np.mean(power), n_steps)
        power = np.clip(power + noise, 100.0, 2500.0)
        
        return power
    
    def _generate_cooling_sequence(self, n_steps: int, scenario_type: str) -> np.ndarray:
        """
        Generate time-varying cooling intensity sequence.
        
        Args:
            n_steps (int): Number of timesteps.
            scenario_type (str): Type of cooling pattern:
                - 'low_constant': constant low cooling
                - 'medium_constant': constant medium cooling
                - 'high_constant': constant high cooling
                - 'step': sudden cooling change
                - 'ramp': gradual increase in cooling
                - 'reactive': reactive to power (rough correlation)
        
        Returns:
            np.ndarray: Cooling intensity sequence [0, 1], shape (n_steps,)
        """
        cooling = np.zeros(n_steps)
        
        if scenario_type == 'low_constant':
            cooling = np.full(n_steps, 0.3)
        
        elif scenario_type == 'medium_constant':
            cooling = np.full(n_steps, 0.6)
        
        elif scenario_type == 'high_constant':
            cooling = np.full(n_steps, 0.9)
        
        elif scenario_type == 'step':
            # Start low, jump to high mid-way
            mid = n_steps // 2
            cooling[:mid] = 0.4
            cooling[mid:] = 0.85
        
        elif scenario_type == 'ramp':
            # Gradual ramp from low to high
            cooling = np.linspace(0.3, 0.95, n_steps)
        
        elif scenario_type == 'reactive':
            # Cooling correlates roughly with power (but delayed)
            # Higher power → higher cooling
            cooling = 0.3 + 0.6 * self.rng.rand(n_steps)
        
        else:
            raise ValueError(f"Unknown cooling scenario_type: {scenario_type}")
        
        # Ensure in valid range [0, 1]
        cooling = np.clip(cooling, 0.0, 1.0)
        
        return cooling
    
    def _generate_scenario(
        self,
        scenario_id: int,
        n_steps: int = 1000,
        power_pattern: str = 'normal_constant',
        cooling_pattern: str = 'medium_constant',
        dt: float = 1.0,
    ) -> List[Dict]:
        """
        Generate a single thermal scenario.
        
        Args:
            scenario_id (int): Unique scenario identifier.
            n_steps (int): Number of timesteps. Default 1000 (~16 min at 1s dt).
            power_pattern (str): Type of power variation.
            cooling_pattern (str): Type of cooling strategy.
            dt (float): Timestep in seconds.
        
        Returns:
            List[Dict]: List of rows (dicts) to be added to DataFrame.
        """
        # Sample random system parameters
        C, hA, T_ambient, T_initial = self._sample_system_parameters()
        
        # Generate workload and cooling sequences
        power_seq = self._generate_power_sequence(n_steps, power_pattern)
        cooling_seq = self._generate_cooling_sequence(n_steps, cooling_pattern)
        
        # Create and run simulator
        model = ThermalModel(C=C, hA=hA, T_ambient=T_ambient, T_initial=T_initial, dt=dt)
        result = model.run_scenario(power_seq, cooling_seq)
        
        # Collect results into list of dicts
        rows = []
        for i in range(n_steps):
            row = {
                'scenario_id': scenario_id,
                'timestep': i,
                'time': result['time'][i],
                'temperature': result['temperature'][i],
                'power': result['power'][i],
                'cooling_intensity': result['cooling_intensity'][i],
                'inlet_temperature': result['inlet_temperature'][i],
                'outlet_temperature': result['outlet_temperature'][i],
                'heat_removal': result['heat_removal'][i],
                # Scenario parameters
                'C': C,
                'hA': hA,
                'T_ambient': T_ambient,
                'dt': dt,
            }
            rows.append(row)
        
        # Store scenario metadata
        self.scenarios.append({
            'scenario_id': scenario_id,
            'power_pattern': power_pattern,
            'cooling_pattern': cooling_pattern,
            'C': C,
            'hA': hA,
            'T_ambient': T_ambient,
            'T_initial': T_initial,
            'n_steps': n_steps,
            'dt': dt,
        })
        
        return rows
    
    def generate(
        self,
        n_scenarios: int = 100,
        n_steps_per_scenario: int = 1000,
        dt: float = 1.0,
    ) -> pd.DataFrame:
        """
        Generate a full synthetic dataset with multiple scenarios.
        
        Args:
            n_scenarios (int): Number of independent scenarios to generate. Default 100.
            n_steps_per_scenario (int): Timesteps per scenario. Default 1000.
            dt (float): Timestep in seconds. Default 1.0.
        
        Returns:
            pd.DataFrame: Dataset with columns:
                - scenario_id, timestep, time
                - temperature, power, cooling_intensity, inlet_temperature, outlet_temperature, heat_removal
                - C, hA, T_ambient, dt
        """
        self.scenarios = []
        self.data_rows = []
        
        # Define available patterns to cycle through
        power_patterns = [
            'low_constant', 'normal_constant', 'high_constant',
            'step', 'spike', 'ramp', 'periodic'
        ]
        cooling_patterns = [
            'low_constant', 'medium_constant', 'high_constant',
            'step', 'ramp', 'reactive'
        ]
        
        for scenario_id in range(n_scenarios):
            # Cycle through patterns for variety
            power_pattern = power_patterns[scenario_id % len(power_patterns)]
            cooling_pattern = cooling_patterns[scenario_id % len(cooling_patterns)]
            
            rows = self._generate_scenario(
                scenario_id=scenario_id,
                n_steps=n_steps_per_scenario,
                power_pattern=power_pattern,
                cooling_pattern=cooling_pattern,
                dt=dt,
            )
            self.data_rows.extend(rows)
        
        # Convert to DataFrame
        df = pd.DataFrame(self.data_rows)
        
        return df
    
    def save(self, df: pd.DataFrame, output_path: str = 'data/synthetic/thermal_dataset.csv'):
        """
        Save the dataset to CSV.
        
        Args:
            df (pd.DataFrame): Dataset to save.
            output_path (str): Path where to save the CSV file.
        """
        # Create parent directory if it doesn't exist
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        print(f"Dataset saved to {output_path}")
        print(f"  - Scenarios: {df['scenario_id'].nunique()}")
        print(f"  - Total rows: {len(df)}")
        print(f"  - Columns: {', '.join(df.columns)}")
    
    def get_scenario_metadata(self) -> pd.DataFrame:
        """
        Return metadata about generated scenarios.
        
        Returns:
            pd.DataFrame: Scenario metadata.
        """
        return pd.DataFrame(self.scenarios)


def main():
    """Command-line interface for dataset generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate physics-consistent synthetic thermal dataset'
    )
    parser.add_argument(
        '--n-scenarios',
        type=int,
        default=100,
        help='Number of scenarios to generate (default 100)'
    )
    parser.add_argument(
        '--n-steps',
        type=int,
        default=1000,
        help='Timesteps per scenario (default 1000)'
    )
    parser.add_argument(
        '--dt',
        type=float,
        default=1.0,
        help='Timestep in seconds (default 1.0)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/synthetic/thermal_dataset.csv',
        help='Output CSV path (default data/synthetic/thermal_dataset.csv)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default 42)'
    )
    
    args = parser.parse_args()
    
    print(f"Generating synthetic thermal dataset...")
    print(f"  Scenarios: {args.n_scenarios}")
    print(f"  Steps per scenario: {args.n_steps}")
    print(f"  Timestep: {args.dt} s")
    print(f"  Seed: {args.seed}")
    
    generator = ThermalDatasetGenerator(seed=args.seed)
    df = generator.generate(
        n_scenarios=args.n_scenarios,
        n_steps_per_scenario=args.n_steps,
        dt=args.dt,
    )
    
    generator.save(df, args.output)
    
    # Print summary statistics
    print("\nDataset summary:")
    print(df.describe())
    
    print("\nScenario breakdown:")
    print(df.groupby('scenario_id')[['power', 'cooling_intensity', 'temperature']].agg(['min', 'max', 'mean']))


if __name__ == '__main__':
    main()
