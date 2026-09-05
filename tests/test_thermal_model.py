"""
Unit tests for the lumped-parameter thermal model.

Tests verify:
- Simulator runs without errors
- Physical behavior is correct (increasing power → increasing temp, etc.)
- Numerical stability and convergence properties
- Parameter validation
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simulator.thermal_model import ThermalModel


class TestParameterValidation:
    """Test that model validates input parameters correctly."""
    
    def test_positive_capacitance_required(self):
        """Test that C > 0 is enforced."""
        with pytest.raises(ValueError, match="Thermal capacitance C must be > 0"):
            ThermalModel(C=0, hA=100.0, T_ambient=293.15, T_initial=298.15)
        
        with pytest.raises(ValueError, match="Thermal capacitance C must be > 0"):
            ThermalModel(C=-100, hA=100.0, T_ambient=293.15, T_initial=298.15)
    
    def test_positive_heat_removal_required(self):
        """Test that hA > 0 is enforced."""
        with pytest.raises(ValueError, match="Heat removal coefficient hA must be > 0"):
            ThermalModel(C=5000, hA=0, T_ambient=293.15, T_initial=298.15)
        
        with pytest.raises(ValueError, match="Heat removal coefficient hA must be > 0"):
            ThermalModel(C=5000, hA=-50, T_ambient=293.15, T_initial=298.15)
    
    def test_valid_ambient_temperature(self):
        """Test that T_ambient must be > 0 and finite."""
        with pytest.raises(ValueError, match="Ambient temperature T_ambient must be > 0 and finite"):
            ThermalModel(C=5000, hA=100, T_ambient=0, T_initial=298.15)
        
        with pytest.raises(ValueError, match="Ambient temperature T_ambient must be > 0 and finite"):
            ThermalModel(C=5000, hA=100, T_ambient=-50, T_initial=298.15)
        
        with pytest.raises(ValueError, match="Ambient temperature T_ambient must be > 0 and finite"):
            ThermalModel(C=5000, hA=100, T_ambient=np.inf, T_initial=298.15)
    
    def test_valid_initial_temperature(self):
        """Test that T_initial must be > 0 and finite."""
        with pytest.raises(ValueError, match="Initial temperature T_initial must be > 0 and finite"):
            ThermalModel(C=5000, hA=100, T_ambient=293.15, T_initial=0)
        
        with pytest.raises(ValueError, match="Initial temperature T_initial must be > 0 and finite"):
            ThermalModel(C=5000, hA=100, T_ambient=293.15, T_initial=-100)
        
        with pytest.raises(ValueError, match="Initial temperature T_initial must be > 0 and finite"):
            ThermalModel(C=5000, hA=100, T_ambient=293.15, T_initial=np.nan)
    
    def test_positive_timestep_required(self):
        """Test that dt > 0 is enforced."""
        with pytest.raises(ValueError, match="Timestep dt must be > 0"):
            ThermalModel(C=5000, hA=100, T_ambient=293.15, T_initial=298.15, dt=0)
        
        with pytest.raises(ValueError, match="Timestep dt must be > 0"):
            ThermalModel(C=5000, hA=100, T_ambient=293.15, T_initial=298.15, dt=-1.0)


class TestInitialization:
    """Test basic initialization and property access."""
    
    def setup_method(self):
        """Set up a standard thermal model for testing."""
        self.C = 5000.0
        self.hA = 100.0
        self.T_ambient = 293.15
        self.T_initial = 298.15
        self.dt = 1.0
        
        self.model = ThermalModel(
            C=self.C,
            hA=self.hA,
            T_ambient=self.T_ambient,
            T_initial=self.T_initial,
            dt=self.dt,
        )
    
    def test_parameters_stored_correctly(self):
        """Test that initialization stores parameters correctly."""
        assert self.model.C == self.C
        assert self.model.hA == self.hA
        assert self.model.T_ambient == self.T_ambient
        assert self.model.T == self.T_initial
        assert self.model.dt == self.dt
    
    def test_initial_temperature_in_kelvin(self):
        """Test that initial temperature is correct in Kelvin."""
        assert self.model.current_temperature == self.T_initial
    
    def test_temperature_conversion_to_celsius(self):
        """Test K to °C conversion."""
        expected_celsius = self.T_initial - 273.15
        assert abs(self.model.current_temperature_celsius - expected_celsius) < 1e-10


class TestPhysicalBehavior:
    """Test that simulated physics behaves correctly."""
    
    def setup_method(self):
        """Set up model for physics tests."""
        self.model = ThermalModel(
            C=5000.0,
            hA=100.0,
            T_ambient=293.15,
            T_initial=298.15,
            dt=0.1,  # Small timestep for stability
        )
    
    def test_zero_power_temperature_decay(self):
        """
        Test that temperature decays toward ambient with P = 0.
        
        With P = 0:
            dT/dt = -hA/C * (T - T_ambient)
        
        Temperature should exponentially decay to T_ambient.
        """
        T_initial = self.model.current_temperature
        assert T_initial > self.model.T_ambient
        
        # Run with zero power for many steps
        power = 0.0
        cooling_intensity = 1.0  # Max cooling to speed up decay
        
        n_steps = 10000
        for _ in range(n_steps):
            self.model.step(power, cooling_intensity)
        
        T_final = self.model.current_temperature
        # Temperature should be much closer to ambient
        assert T_final < T_initial
        # Should be very close to effective ambient (which includes cooling)
        T_ambient_eff = self.model.T_ambient - 10.0  # Cooling boost
        assert abs(T_final - T_ambient_eff) < 0.5, \
            f"After {n_steps} steps with P=0, T should decay to {T_ambient_eff:.2f} K, " \
            f"got {T_final:.2f} K"
    
    def test_higher_power_higher_steady_state(self):
        """
        Test that increasing server power increases steady-state temperature.
        
        Two scenarios with different power levels, identical cooling.
        Higher power should result in higher final temperature.
        """
        # Scenario 1: low power
        model_low = ThermalModel(C=5000.0, hA=100.0, T_ambient=293.15, T_initial=298.15, dt=0.1)
        power_low = 500.0
        for _ in range(1000):
            model_low.step(power_low, cooling_intensity=0.5)
        T_low = model_low.current_temperature
        
        # Scenario 2: high power
        model_high = ThermalModel(C=5000.0, hA=100.0, T_ambient=293.15, T_initial=298.15, dt=0.1)
        power_high = 2000.0
        for _ in range(1000):
            model_high.step(power_high, cooling_intensity=0.5)
        T_high = model_high.current_temperature
        
        assert T_high > T_low, \
            f"Higher power ({power_high} W) should produce higher temp. " \
            f"T_low={T_low:.2f}, T_high={T_high:.2f}"
    
    def test_higher_cooling_lower_steady_state(self):
        """
        Test that increasing cooling intensity decreases steady-state temperature.
        
        Two scenarios with different cooling levels, identical power.
        Higher cooling should result in lower final temperature.
        """
        # Scenario 1: no cooling
        model_no_cool = ThermalModel(C=5000.0, hA=100.0, T_ambient=293.15, T_initial=298.15, dt=0.1)
        power = 1000.0
        for _ in range(1000):
            model_no_cool.step(power, cooling_intensity=0.0)
        T_no_cool = model_no_cool.current_temperature
        
        # Scenario 2: max cooling
        model_max_cool = ThermalModel(C=5000.0, hA=100.0, T_ambient=293.15, T_initial=298.15, dt=0.1)
        for _ in range(1000):
            model_max_cool.step(power, cooling_intensity=1.0)
        T_max_cool = model_max_cool.current_temperature
        
        assert T_max_cool < T_no_cool, \
            f"Higher cooling should produce lower temp. " \
            f"T_no_cool={T_no_cool:.2f}, T_max_cool={T_max_cool:.2f}"
    
    def test_smooth_temperature_evolution(self):
        """
        Test that temperature evolves smoothly (no large jumps).
        
        With a small timestep, temperature change per step should be smooth.
        """
        power_sequence = np.full(100, 1000.0)
        cooling_sequence = np.zeros(100)
        
        result = self.model.run_scenario(power_sequence, cooling_sequence)
        temperatures = result['temperature']
        
        # Check that changes are smooth
        temp_differences = np.diff(temperatures)
        max_change_per_step = np.max(np.abs(temp_differences))
        
        # With dt=0.1 and C=5000, max change should be small
        assert max_change_per_step < 0.1, \
            f"Temperature changed by {max_change_per_step:.4f} K in one step; " \
            f"should be smooth"
    
    def test_increasing_ambient_increases_temperature(self):
        """
        Test that higher ambient temperature results in higher steady-state.
        
        With fixed power and cooling, higher ambient should lead to higher steady-state.
        """
        # Scenario 1: low ambient
        model_low_amb = ThermalModel(C=5000.0, hA=100.0, T_ambient=288.15, T_initial=293.15, dt=0.1)
        power = 1000.0
        for _ in range(1000):
            model_low_amb.step(power, cooling_intensity=0.0)
        T_low_amb = model_low_amb.current_temperature
        
        # Scenario 2: high ambient
        model_high_amb = ThermalModel(C=5000.0, hA=100.0, T_ambient=303.15, T_initial=293.15, dt=0.1)
        for _ in range(1000):
            model_high_amb.step(power, cooling_intensity=0.0)
        T_high_amb = model_high_amb.current_temperature
        
        assert T_high_amb > T_low_amb, \
            f"Higher ambient should produce higher temp. " \
            f"T_low_amb={T_low_amb:.2f}, T_high_amb={T_high_amb:.2f}"


class TestNumericalStability:
    """Test numerical stability properties of forward Euler integration."""
    
    def test_stability_check_large_timestep(self, capsys):
        """
        Test that large timestep triggers stability warning.
        
        Stability limit for forward Euler: dt < 2*C/hA.
        """
        # Set dt >= stability limit
        C = 1000.0
        hA = 1000.0
        stability_limit = 2.0 * C / hA  # = 2.0
        large_dt = 2.5
        
        model = ThermalModel(
            C=C,
            hA=hA,
            T_ambient=293.15,
            T_initial=298.15,
            dt=large_dt,
        )
        
        captured = capsys.readouterr()
        assert "WARNING" in captured.out or "stability" in captured.out.lower(), \
            "Expected stability warning for large timestep"
    
    def test_stability_check_small_timestep(self, capsys):
        """
        Test that small timestep does not trigger warning.
        """
        # Set dt well below stability limit
        C = 5000.0
        hA = 100.0
        stability_limit = 2.0 * C / hA  # = 100.0
        small_dt = 1.0  # << 100.0
        
        model = ThermalModel(
            C=C,
            hA=hA,
            T_ambient=293.15,
            T_initial=298.15,
            dt=small_dt,
        )
        
        captured = capsys.readouterr()
        # Should not have stability warning
        assert "WARNING" not in captured.out, \
            "Should not warn for small timestep"
    
    def test_energy_balance_accuracy(self):
        """
        Test that energy balance is approximately conserved at steady state.
        
        At steady state: dT/dt = 0 → power = hA * (T - T_ambient_eff)
        Verify this holds after many steps.
        """
        model = ThermalModel(C=5000.0, hA=100.0, T_ambient=293.15, T_initial=298.15, dt=0.1)
        
        power = 1000.0
        cooling_intensity = 0.5
        
        # Run to steady state
        for _ in range(5000):
            model.step(power, cooling_intensity)
        
        # At steady state, power ≈ hA * (T - T_ambient_eff)
        T_ambient_eff = model.T_ambient - cooling_intensity * 10.0
        heat_removal_at_ss = model.hA * (model.current_temperature - T_ambient_eff)
        
        # Should be approximately equal
        error_fraction = abs(heat_removal_at_ss - power) / power
        assert error_fraction < 0.02, \
            f"Energy balance error: {error_fraction*100:.1f}%. " \
            f"Power={power}, Heat removal={heat_removal_at_ss:.1f}"


class TestInputValidationStep:
    """Test step() method input validation."""
    
    def setup_method(self):
        """Set up model."""
        self.model = ThermalModel(
            C=5000.0,
            hA=100.0,
            T_ambient=293.15,
            T_initial=298.15,
            dt=1.0,
        )
    
    def test_negative_power_rejected(self):
        """Test that negative power raises ValueError."""
        with pytest.raises(ValueError, match="Power must be >= 0"):
            self.model.step(power=-100.0, cooling_intensity=0.5)
    
    def test_cooling_intensity_bounds(self):
        """Test that cooling_intensity must be in [0, 1]."""
        with pytest.raises(ValueError, match="cooling_intensity must be in"):
            self.model.step(power=1000.0, cooling_intensity=1.5)
        
        with pytest.raises(ValueError, match="cooling_intensity must be in"):
            self.model.step(power=1000.0, cooling_intensity=-0.1)
    
    def test_valid_boundary_values(self):
        """Test that boundary values (0 and 1) are accepted."""
        # Should not raise
        self.model.step(power=1000.0, cooling_intensity=0.0)
        self.model.step(power=1000.0, cooling_intensity=1.0)
    
    def test_invalid_ambient_update(self):
        """Test that invalid T_ambient update is rejected."""
        with pytest.raises(ValueError, match="Updated T_ambient must be > 0 and finite"):
            self.model.step(power=1000.0, cooling_intensity=0.5, T_ambient=-100)
        
        with pytest.raises(ValueError, match="Updated T_ambient must be > 0 and finite"):
            self.model.step(power=1000.0, cooling_intensity=0.5, T_ambient=np.nan)


class TestScenarioGeneration:
    """Test run_scenario() method."""
    
    def setup_method(self):
        """Set up model."""
        self.model = ThermalModel(
            C=5000.0,
            hA=100.0,
            T_ambient=293.15,
            T_initial=298.15,
            dt=1.0,
        )
    
    def test_scenario_returns_all_fields(self):
        """Test that run_scenario returns all expected data fields."""
        power = np.array([500.0, 1000.0, 1500.0])
        cooling = np.array([0.0, 0.5, 1.0])
        
        result = self.model.run_scenario(power, cooling)
        
        expected_keys = {
            'time', 'temperature', 'power', 'cooling_intensity',
            'inlet_temperature', 'outlet_temperature', 'heat_removal'
        }
        assert set(result.keys()) == expected_keys
    
    def test_scenario_output_length(self):
        """Test that output has same length as input sequences."""
        n = 100
        power = np.random.rand(n) * 2000.0
        cooling = np.random.rand(n)
        
        result = self.model.run_scenario(power, cooling)
        
        for key, value in result.items():
            assert len(value) == n, \
                f"Key '{key}' has length {len(value)}, expected {n}"
    
    def test_scenario_with_varying_ambient(self):
        """Test run_scenario with time-varying ambient temperature."""
        power = np.array([1000.0, 1000.0, 1000.0])
        cooling = np.array([0.0, 0.0, 0.0])
        # Ambient increases from 293 K to 303 K (20°C to 30°C)
        T_ambient = np.array([293.15, 298.15, 303.15])
        
        result = self.model.run_scenario(power, cooling, T_ambient)
        
        # As ambient increases, outlet temperature should increase
        assert result['outlet_temperature'][-1] > result['outlet_temperature'][0]
    
    def test_scenario_length_mismatch_error(self):
        """Test that mismatched sequence lengths raise AssertionError."""
        power = np.array([1000.0, 1000.0])
        cooling = np.array([0.0, 0.5, 1.0])  # Different length
        
        with pytest.raises(AssertionError):
            self.model.run_scenario(power, cooling)
    
    def test_scenario_T_ambient_length_mismatch(self):
        """Test that mismatched T_ambient length raises AssertionError."""
        power = np.array([1000.0, 1000.0, 1000.0])
        cooling = np.array([0.0, 0.5, 1.0])
        T_ambient = np.array([293.15, 298.15])  # Different length
        
        with pytest.raises(AssertionError):
            self.model.run_scenario(power, cooling, T_ambient)


class TestReset:
    """Test reset() method."""
    
    def setup_method(self):
        """Set up model."""
        self.model = ThermalModel(
            C=5000.0,
            hA=100.0,
            T_ambient=293.15,
            T_initial=298.15,
            dt=1.0,
        )
    
    def test_reset_clears_temperature(self):
        """Test that reset sets temperature to new value."""
        # Run some steps
        for _ in range(10):
            self.model.step(power=1000.0, cooling_intensity=0.5)
        
        assert self.model.current_temperature != 298.15
        
        # Reset
        self.model.reset(T_initial=298.15)
        assert self.model.current_temperature == 298.15
    
    def test_reset_clears_history(self):
        """Test that reset clears the history."""
        # Run some steps
        result = self.model.run_scenario(
            np.array([500.0, 1000.0]),
            np.array([0.0, 0.5])
        )
        assert len(result['temperature']) > 0
        
        # Reset
        self.model.reset(T_initial=298.15)
        assert len(self.model.history['temperature']) == 0
    
    def test_reset_invalid_temperature(self):
        """Test that reset rejects invalid temperature."""
        with pytest.raises(ValueError, match="T_initial must be > 0 and finite"):
            self.model.reset(T_initial=0)
        
        with pytest.raises(ValueError, match="T_initial must be > 0 and finite"):
            self.model.reset(T_initial=np.inf)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
