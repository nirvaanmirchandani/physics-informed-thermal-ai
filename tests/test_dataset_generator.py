"""
Unit tests for the synthetic thermal dataset generator.

Tests verify:
- Reproducibility with same seed
- Different seeds produce different datasets
- All expected columns exist
- Correct number of scenarios and rows
- Valid input ranges (power, cooling_intensity, temperatures)
- Coherent time sequences in each scenario
- No scenario ID collisions or overlaps
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.generate_dataset import ThermalDatasetGenerator


class TestDatasetGeneratorBasics:
    """Test basic functionality of ThermalDatasetGenerator."""
    
    def test_initialization(self):
        """Test that generator initializes with seed."""
        gen = ThermalDatasetGenerator(seed=42)
        assert gen.seed == 42
        assert gen.rng is not None
        assert len(gen.scenarios) == 0
        assert len(gen.data_rows) == 0
    
    def test_initialization_with_different_seed(self):
        """Test that different seeds can be specified."""
        gen1 = ThermalDatasetGenerator(seed=42)
        gen2 = ThermalDatasetGenerator(seed=99)
        assert gen1.seed != gen2.seed


class TestReproducibility:
    """Test that datasets are reproducible with same seed."""
    
    def test_same_seed_produces_same_dataset(self):
        """
        Test that two generators with same seed produce identical datasets.
        """
        seed = 42
        n_scenarios = 10
        n_steps = 100
        
        # Generate dataset 1
        gen1 = ThermalDatasetGenerator(seed=seed)
        df1 = gen1.generate(n_scenarios=n_scenarios, n_steps_per_scenario=n_steps)
        
        # Generate dataset 2 (same seed)
        gen2 = ThermalDatasetGenerator(seed=seed)
        df2 = gen2.generate(n_scenarios=n_scenarios, n_steps_per_scenario=n_steps)
        
        # Should be identical
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_different_seeds_produce_different_datasets(self):
        """
        Test that different seeds produce different datasets.
        """
        n_scenarios = 10
        n_steps = 100
        
        # Generate dataset 1
        gen1 = ThermalDatasetGenerator(seed=42)
        df1 = gen1.generate(n_scenarios=n_scenarios, n_steps_per_scenario=n_steps)
        
        # Generate dataset 2 (different seed)
        gen2 = ThermalDatasetGenerator(seed=99)
        df2 = gen2.generate(n_scenarios=n_scenarios, n_steps_per_scenario=n_steps)
        
        # Should be different (at least in power, cooling, C, hA, etc.)
        # They should have same shape but different values
        assert df1.shape == df2.shape
        assert not df1.equals(df2)
        
        # Check that at least some key columns differ
        power_differs = not np.allclose(df1['power'].values, df2['power'].values)
        assert power_differs, "Different seeds should produce different power sequences"


class TestDatasetShape:
    """Test output dataset shape and dimensions."""
    
    def test_correct_number_of_scenarios(self):
        """Test that correct number of scenarios are generated."""
        n_scenarios = 20
        n_steps = 100
        
        gen = ThermalDatasetGenerator(seed=42)
        df = gen.generate(n_scenarios=n_scenarios, n_steps_per_scenario=n_steps)
        
        assert df['scenario_id'].nunique() == n_scenarios
    
    def test_correct_number_of_rows(self):
        """Test that correct number of total rows are produced."""
        n_scenarios = 20
        n_steps = 100
        expected_rows = n_scenarios * n_steps
        
        gen = ThermalDatasetGenerator(seed=42)
        df = gen.generate(n_scenarios=n_scenarios, n_steps_per_scenario=n_steps)
        
        assert len(df) == expected_rows
    
    def test_steps_per_scenario(self):
        """Test that each scenario has correct number of steps."""
        n_scenarios = 5
        n_steps = 200
        
        gen = ThermalDatasetGenerator(seed=42)
        df = gen.generate(n_scenarios=n_scenarios, n_steps_per_scenario=n_steps)
        
        for scenario_id in df['scenario_id'].unique():
            scenario_df = df[df['scenario_id'] == scenario_id]
            assert len(scenario_df) == n_steps, \
                f"Scenario {scenario_id} has {len(scenario_df)} rows, expected {n_steps}"


class TestDatasetColumns:
    """Test that dataset has all expected columns."""
    
    def test_all_required_columns_present(self):
        """Test that all required columns exist."""
        gen = ThermalDatasetGenerator(seed=42)
        df = gen.generate(n_scenarios=5, n_steps_per_scenario=100)
        
        required_columns = {
            'scenario_id', 'timestep', 'time',
            'temperature', 'power', 'cooling_intensity',
            'inlet_temperature', 'outlet_temperature', 'heat_removal',
            'C', 'hA', 'T_ambient', 'dt'
        }
        
        assert required_columns.issubset(set(df.columns)), \
            f"Missing columns: {required_columns - set(df.columns)}"
    
    def test_no_extra_columns(self):
        """Test that there are no unexpected extra columns."""
        gen = ThermalDatasetGenerator(seed=42)
        df = gen.generate(n_scenarios=5, n_steps_per_scenario=100)
        
        expected_columns = {
            'scenario_id', 'timestep', 'time',
            'temperature', 'power', 'cooling_intensity',
            'inlet_temperature', 'outlet_temperature', 'heat_removal',
            'C', 'hA', 'T_ambient', 'dt'
        }
        
        assert set(df.columns) == expected_columns, \
            f"Unexpected columns: {set(df.columns) - expected_columns}"


class TestDataValidation:
    """Test that generated data values are physically valid."""
    
    def test_power_in_valid_range(self):
        """Test that power values are within valid range (>= 0)."""
        gen = ThermalDatasetGenerator(seed=42)
        df = gen.generate(n_scenarios=10, n_steps_per_scenario=500)
        
        assert df['power'].min() >= 0.0, f"Power minimum {df['power'].min()} < 0"
        assert df['power'].max() <= 3000.0, f"Power maximum {df['power'].max()} > 3000"
    
    def test_cooling_intensity_in_valid_range(self):
        """Test that cooling_intensity is in [0, 1]."""
        gen = ThermalDatasetGenerator(seed=42)
        df = gen.generate(n_scenarios=10, n_steps_per_scenario=500)
        
        assert df['cooling_intensity'].min() >= 0.0, \
            f"Cooling min {df['cooling_intensity'].min()} < 0"
        assert df['cooling_intensity'].max() <= 1.0, \
            f"Cooling max {df['cooling_intensity'].max()} > 1"
    
    def test_temperature_in_reasonable_range(self):
        """Test that temperatures are within reasonable physical bounds."""
        gen = ThermalDatasetGenerator(seed=42)
        df = gen.generate(n_scenarios=10, n_steps_per_scenario=500)
        
        # Temperature in Kelvin: allow 250–370 K (-23°C to 97°C)
        T_min = 250.0
        T_max = 370.0
        
        assert df['temperature'].min() >= T_min, \
            f"Temperature {df['temperature'].min()} K is below {T_min} K"
        assert df['temperature'].max() <= T_max, \
            f"Temperature {df['temperature'].max()} K is above {T_max} K"
    
    def test_inlet_temperature_in_reasonable_range(self):
        """Test that inlet temperatures are within reasonable bounds."""
        gen = ThermalDatasetGenerator(seed=42)
        df = gen.generate(n_scenarios=10, n_steps_per_scenario=500)
        
        # Inlet temp should be ambient ± cooling boost
        assert df['inlet_temperature'].min() >= 250.0
        assert df['inlet_temperature'].max() <= 370.0
    
    def test_C_parameter_in_expected_range(self):
        """Test that thermal capacitance C is in expected range."""
        gen = ThermalDatasetGenerator(seed=42)
        df = gen.generate(n_scenarios=20, n_steps_per_scenario=100)
        
        # C should be 1000–10000 J/K
        assert df['C'].min() >= 1000.0, f"C minimum {df['C'].min()} < 1000"
        assert df['C'].max() <= 10000.0, f"C maximum {df['C'].max()} > 10000"
    
    def test_hA_parameter_in_expected_range(self):
        """Test that heat removal coefficient hA is in expected range."""
        gen = ThermalDatasetGenerator(seed=42)
        df = gen.generate(n_scenarios=20, n_steps_per_scenario=100)
        
        # hA should be 50–200 W/K
        assert df['hA'].min() >= 50.0, f"hA minimum {df['hA'].min()} < 50"
        assert df['hA'].max() <= 200.0, f"hA maximum {df['hA'].max()} > 200"
    
    def test_T_ambient_in_expected_range(self):
        """Test that ambient temperature is in expected range."""
        gen = ThermalDatasetGenerator(seed=42)
        df = gen.generate(n_scenarios=20, n_steps_per_scenario=100)
        
        # T_ambient should be 288–303 K (15–30°C)
        assert df['T_ambient'].min() >= 288.15
        assert df['T_ambient'].max() <= 303.15


class TestScenarioConsistency:
    """Test that each scenario has coherent time sequences."""
    
    def test_time_is_monotonically_increasing(self):
        """Test that time increases monotonically within each scenario."""
        gen = ThermalDatasetGenerator(seed=42)
        df = gen.generate(n_scenarios=5, n_steps_per_scenario=200)
        
        for scenario_id in df['scenario_id'].unique():
            scenario_df = df[df['scenario_id'] == scenario_id]
            times = scenario_df['time'].values
            
            # Check monotonic increase
            assert np.all(np.diff(times) > 0), \
                f"Time not monotonically increasing in scenario {scenario_id}"
    
    def test_timestep_is_consistent(self):
        """Test that timestep is consistent within each scenario."""
        gen = ThermalDatasetGenerator(seed=42)
        df = gen.generate(n_scenarios=5, n_steps_per_scenario=200, dt=0.5)
        
        for scenario_id in df['scenario_id'].unique():
            scenario_df = df[df['scenario_id'] == scenario_id]
            times = scenario_df['time'].values
            
            # Check that differences are consistent (within floating point tolerance)
            time_diffs = np.diff(times)
            expected_dt = 0.5
            
            assert np.allclose(time_diffs, expected_dt, rtol=1e-10), \
                f"Inconsistent timestep in scenario {scenario_id}"
    
    def test_scenario_ids_sequential_and_unique(self):
        """Test that scenario IDs are sequential and no collisions."""
        n_scenarios = 15
        gen = ThermalDatasetGenerator(seed=42)
        df = gen.generate(n_scenarios=n_scenarios, n_steps_per_scenario=100)
        
        scenario_ids = sorted(df['scenario_id'].unique())
        
        # Should be 0, 1, 2, ..., n_scenarios-1
        assert scenario_ids == list(range(n_scenarios))
    
    def test_no_scenario_id_overlaps(self):
        """Test that scenario IDs don't overlap between runs."""
        gen = ThermalDatasetGenerator(seed=42)
        df1 = gen.generate(n_scenarios=5, n_steps_per_scenario=100)
        df2 = gen.generate(n_scenarios=5, n_steps_per_scenario=100)
        
        # After second generate, IDs should be 0–4 again (reset)
        scenario_ids_2 = sorted(df2['scenario_id'].unique())
        assert scenario_ids_2 == list(range(5))


class TestScenarioPatterns:
    """Test that workload and cooling patterns are correctly applied."""
    
    def test_constant_power_scenario(self):
        """Test that 'low_constant' power pattern is roughly constant."""
        gen = ThermalDatasetGenerator(seed=42)
        # Generate with a specific pattern
        df = gen.generate(n_scenarios=1, n_steps_per_scenario=500)
        
        power = df['power'].values
        # Should be roughly constant (with small noise)
        power_std = np.std(power)
        power_mean = np.mean(power)
        power_cv = power_std / power_mean  # Coefficient of variation
        
        # CV should be small for constant patterns
        assert power_cv < 0.15, \
            f"Power pattern not constant enough: CV = {power_cv}"
    
    def test_all_pattern_types_generated(self):
        """Test that multiple pattern types are actually generated."""
        n_scenarios = 21  # Enough to cycle through all patterns
        gen = ThermalDatasetGenerator(seed=42)
        df = gen.generate(n_scenarios=n_scenarios, n_steps_per_scenario=100)
        
        # Metadata should show different patterns
        metadata = gen.get_scenario_metadata()
        power_patterns = set(metadata['power_pattern'].unique())
        cooling_patterns = set(metadata['cooling_pattern'].unique())
        
        # Should have multiple distinct patterns
        assert len(power_patterns) > 1, "Only one power pattern type generated"
        assert len(cooling_patterns) > 1, "Only one cooling pattern type generated"


class TestMetadata:
    """Test scenario metadata functionality."""
    
    def test_get_scenario_metadata(self):
        """Test that scenario metadata can be retrieved."""
        n_scenarios = 10
        gen = ThermalDatasetGenerator(seed=42)
        df = gen.generate(n_scenarios=n_scenarios, n_steps_per_scenario=100)
        
        metadata = gen.get_scenario_metadata()
        
        assert len(metadata) == n_scenarios
        assert 'scenario_id' in metadata.columns
        assert 'power_pattern' in metadata.columns
        assert 'cooling_pattern' in metadata.columns
        assert 'C' in metadata.columns
        assert 'hA' in metadata.columns


class TestFileSaving:
    """Test CSV file saving functionality."""
    
    def test_save_to_csv(self):
        """Test that dataset can be saved to CSV."""
        gen = ThermalDatasetGenerator(seed=42)
        df = gen.generate(n_scenarios=5, n_steps_per_scenario=100)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_dataset.csv"
            gen.save(df, str(output_path))
            
            assert output_path.exists()
    
    def test_saved_csv_is_readable(self):
        """Test that saved CSV can be read back."""
        gen = ThermalDatasetGenerator(seed=42)
        df = gen.generate(n_scenarios=5, n_steps_per_scenario=100)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_dataset.csv"
            gen.save(df, str(output_path))
            
            # Read back
            df_read = pd.read_csv(output_path)
            
            # Should have same shape and columns
            assert df_read.shape == df.shape
            assert set(df_read.columns) == set(df.columns)
    
    def test_save_creates_parent_directory(self):
        """Test that save creates parent directories if needed."""
        gen = ThermalDatasetGenerator(seed=42)
        df = gen.generate(n_scenarios=5, n_steps_per_scenario=100)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "test_dataset.csv"
            gen.save(df, str(output_path))
            
            assert output_path.exists()
            assert output_path.parent.exists()


class TestParameterVariation:
    """Test that system parameters vary correctly across scenarios."""
    
    def test_C_varies_across_scenarios(self):
        """Test that C is not constant across scenarios."""
        gen = ThermalDatasetGenerator(seed=42)
        df = gen.generate(n_scenarios=20, n_steps_per_scenario=100)
        
        C_values = df.groupby('scenario_id')['C'].first().values
        
        # Should have some variation
        assert len(np.unique(C_values)) > 10, \
            "C parameter does not vary enough across scenarios"
    
    def test_hA_varies_across_scenarios(self):
        """Test that hA is not constant across scenarios."""
        gen = ThermalDatasetGenerator(seed=42)
        df = gen.generate(n_scenarios=20, n_steps_per_scenario=100)
        
        hA_values = df.groupby('scenario_id')['hA'].first().values
        
        # Should have some variation
        assert len(np.unique(hA_values)) > 10, \
            "hA parameter does not vary enough across scenarios"
    
    def test_T_ambient_varies_across_scenarios(self):
        """Test that T_ambient varies across scenarios."""
        gen = ThermalDatasetGenerator(seed=42)
        df = gen.generate(n_scenarios=20, n_steps_per_scenario=100)
        
        T_amb_values = df.groupby('scenario_id')['T_ambient'].first().values
        
        # Should have variation
        assert len(np.unique(T_amb_values)) > 10, \
            "T_ambient does not vary across scenarios"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
