# Physics-Informed Thermal AI

A physics-informed machine learning system for thermal prediction and cooling optimization.

## Stage 1: Thermal Simulator (Complete)

This stage implements a **lumped-parameter thermal model** for generating realistic synthetic datacenter thermal data.

### What This Is

A simple but scientifically sound thermal simulator that models a datacenter rack as a single lumped thermal mass. The model solves the energy balance equation:

```
C * dT/dt = P(t) - hA * (T - T_ambient_eff)
```

Where:
- **T** = rack/server temperature (K)
- **P(t)** = server power dissipation (W)
- **hA** = effective heat removal coefficient (W/K)
- **C** = thermal capacitance (J/K)
- **T_ambient_eff** = effective ambient temperature with active cooling (K)

### Why Lumped-Parameter?

Real datacenters have complex 3D thermal fields. However, for an initial ML research project, a lumped model:
- ✅ Captures essential physics (power → heat → cooling → temperature)
- ✅ Is fast enough to generate large datasets
- ✅ Is interpretable (few parameters, clear equations)
- ✅ Can later be replaced with real sensor data via a simple adapter

### Key Modeling Simplifications

1. **Single temperature node** – represents average rack inlet temperature
2. **Linear heat transfer** – Q = hA·ΔT (valid for modest temperature differences)
3. **Constant parameters** – C and hA do not vary with temperature
4. **Cooling simplification** – cooling intensity reduces effective ambient by up to 10 K
   - This is a *modeling choice*, not empirical CRAC performance
   - Real systems depend on mass flow, inlet conditions, control dynamics
5. **No spatial distribution** – no CFD, no 3D mesh (saved for future stages)

### Model Parameters

All parameters are validated for physical reasonableness:

| Parameter | Symbol | Units | Notes |
|-----------|--------|-------|-------|
| Thermal capacitance | C | J/K | Must be > 0. Typical: 1000–10000 for server rack. |
| Heat removal coefficient | hA | W/K | Must be > 0. Typical: 50–200 for passive + active cooling. |
| Ambient temperature | T_ambient | K | Must be > 0 and finite. Typical: 293.15 K (20°C). |
| Timestep | dt | s | Must be > 0. Stability requires dt < 2C/hA. |

### Numerical Method

- **Integration**: Forward Euler (O(dt) error per step)
- **Stability check**: At initialization, warns if dt exceeds stability limit 2C/hA
- **No clamping**: Temperature evolves freely; parameter validation ensures physical bounds

### Code Structure

```
src/
  simulator/
    __init__.py           # Package exports
    thermal_model.py      # Core ThermalModel class

tests/
  test_thermal_model.py   # 28 unit tests covering:
                          # - Parameter validation
                          # - Physical behavior (power, cooling, ambient effects)
                          # - Numerical stability
                          # - Scenario generation

conftest.py              # Pytest configuration
```

### How to Use

#### 1. Single Timestep Simulation

```python
from simulator import ThermalModel
import numpy as np

# Create model
model = ThermalModel(
    C=5000.0,                # J/K (thermal mass)
    hA=100.0,                # W/K (cooling efficiency)
    T_ambient=293.15,        # K (20°C)
    T_initial=298.15,        # K (25°C)
    dt=0.1,                  # s (timestep)
)

# Simulate one step
power = 1000.0             # W (server dissipation)
cooling_intensity = 0.5    # 0-1 (0=no cooling, 1=max cooling)
T_new = model.step(power, cooling_intensity)

print(f"Temperature: {model.current_temperature_celsius:.2f}°C")
```

#### 2. Run a Scenario (Time Series)

```python
# Define time-varying workload and cooling
n_steps = 1000
power_sequence = np.linspace(500, 2000, n_steps)
cooling_sequence = 0.7 * np.ones(n_steps)

# Run simulation
result = model.run_scenario(power_sequence, cooling_sequence)

# result contains:
# - result['time']: elapsed time (s)
# - result['temperature']: rack temperature (K)
# - result['power']: server power (W)
# - result['cooling_intensity']: cooling level (0-1)
# - result['inlet_temperature']: effective ambient (K)
# - result['outlet_temperature']: rack outlet temp (K)
# - result['heat_removal']: convective cooling (W)
```

#### 3. Reset and Reuse

```python
model.reset(T_initial=298.15)  # Reset temperature, clear history
```

### Running Tests

All tests validate **physics correctness** and **numerical stability**, not implementation details.

```bash
# Run all tests
pytest tests/test_thermal_model.py -v

# Run specific test class
pytest tests/test_thermal_model.py::TestPhysicalBehavior -v

# Run with coverage
pytest tests/test_thermal_model.py --cov=src/simulator --cov-report=html
```

#### Test Summary (28 tests)

| Test Class | Count | Purpose |
|------------|-------|---------|
| TestParameterValidation | 5 | Ensure C, hA, T_ambient, T_initial, dt are > 0 and finite |
| TestInitialization | 3 | Parameter storage, K/°C conversion |
| TestPhysicalBehavior | 5 | Zero power → decay, higher power → hotter, higher cooling → cooler, smooth evolution, ambient effects |
| TestNumericalStability | 3 | Euler stability warning, energy balance at steady state |
| TestInputValidationStep | 4 | Reject invalid power, cooling, ambient in step() |
| TestScenarioGeneration | 5 | Output fields, lengths, varying ambient, length mismatch |
| TestReset | 3 | Clear temperature, history, reject invalid T_initial |

**All tests pass** ✅

### Expected Output

#### Example: Constant Power, Varying Cooling

```python
model = ThermalModel(C=5000, hA=100, T_ambient=293.15, T_initial=298.15, dt=0.1)

power = np.full(500, 1000.0)  # Constant 1 kW
cooling = np.concatenate([
    np.zeros(100),      # No cooling
    0.5 * np.ones(100), # 50% cooling
    np.ones(200),       # 100% cooling
])

result = model.run_scenario(power, cooling)

# Expected behavior:
# Phase 1 (t=0-100): Temperature rises (no cooling)
# Phase 2 (t=100-200): Temperature stabilizes higher (partial cooling)
# Phase 3 (t=200-500): Temperature drops then plateaus lower (max cooling)
```

**Physical plausibility checks:**
- ✅ Temperature rises when power > heat removal
- ✅ Temperature decays toward steady state exponentially
- ✅ Increasing cooling reduces steady-state temperature
- ✅ Increasing ambient increases steady-state temperature
- ✅ Energy balance holds: Power ≈ hA·(T - T_amb) at steady state
- ✅ No oscillations or divergence (stable numerical method)

### Dependencies

- **numpy**: Numerical arrays
- **pytest**: Testing framework

No other dependencies (no ML frameworks, no plotting, no CFD solvers yet).

### Next Stage: ML Models (Not Yet Implemented)

The thermal simulator generates realistic synthetic data that will be used to train:

1. **Pure ML baselines** (Dense NN, LSTM)
2. **Physics-informed models** (PINN, Bayesian hybrid)
3. **Predictive models** for cooling optimization

The data format is designed to interface cleanly with ML pipelines (numpy arrays → pandas → PyTorch/TensorFlow).

### Limitations & Future Work

**Current limitations:**
- Single lumped node (no spatial distribution)
- No humidity modeling
- No latent heat / phase changes
- No radiation-dominated regimes
- Cooling boost (10 K) is hardcoded (not empirically derived)

**Future enhancements (Stages 2+):**
- Real sensor data adapter (replace simulator with live datacenter data)
- Multi-zone thermal model (e.g., hot aisle / cold aisle)
- Stochastic workload patterns (Poisson arrivals, burstiness)
- CFD-generated lookup tables (hybrid approach)

### References

- Energy balance: 1st-order ODE, fundamental heat transfer
- Forward Euler stability: standard numerical analysis (CFL condition)
- Lumped-capacity method: widely used in HVAC and thermal management
- Datacenter thermal modeling: see PUE, ASHRAE thermal guidelines

---

**Version**: 0.1.0  
**Status**: Stage 1 complete, ready for synthetic data generation  
**Last Updated**: 2026-09-05
