
# Descriptor System Control: DeePC & MPC & ID + MPC

This repository contains MATLAB implementations of **Data-enabled Predictive Control (DeePC)** ， **Model Predictive Control (MPC)** and **System Identification + Modelabetic Control (ID + MPC)** for **discrete-time descriptor systems**, with applications to two independent case studies:

- **Power System**
- **Water Network**

The project focuses on control design and performance comparison between **DeePC**, **MPC**, and **ID + MPC**.

---

## 📁 Repository Structure

### Control Analysis Functions
- `quasi_weierstrass.m`  
  Transform a descriptor system (E,A,B,C) into **quasi-Weierstrass$\beta$ form**.

- `nilpotent_index.m`  
  Compute the **nilpotent index** of a nilpotent matrix.

- `check_R_controllability.m`  
  Check **R-controllability** of a descriptor system (E,A,B).

- `check_R_observability.m`  
  Check **R-observability** of a descriptor system (E,A,C).

- `identify_descriptor_system.m`  
  Identify descriptor system matrices from input-output data.
  
- `identify_descriptor_system_v2.m`  
  Descriptor system identification under output measurement noise

- `hankel.m`  
  Construct Hankel matrices.
---

### Optimization Functions
- `deepc_optimization.m`  
  Solve the **DeePC optimization problem** using CVX and Hankel matrices.  
  Supports noisy and noise-free formulations with terminal constraints.

- `mpc_optimization.m`  
  Solve the **MPC optimization problem** for descriptor systems with input constraints.

---

### Main Simulation Scripts
- `main_power_system.m` & `main_power_system_noise.m`  
  Main simulation for the **power system** case study.  
  Implements DeePC, MPC, and ID + MPC controllers and compares their performance under ideal (noise-free) conditions.  
  The noise version (`main_power_system_noise.m`) adds measurement noise to the training/output data and evaluates the robustness of DeePC and ID + MPC.

- `main_water_network.m` & `main_water_network_noise.m`  
  Main simulation for the **water network** case study.  
  Constructs descriptor system matrices from EPANET data and applies DeePC, MPC, and ID + MPC controllers.  
  The noise version (`main_water_network_noise.m`) contaminates the training/output data with additive noise and compares only DeePC and ID + MPC to assess robustness under noisy measurements.


---

## Parameter Configuration Guide

### Basic Simulation Parameters

| Parameter | Description | Impact |
|----------|------------|--------|
| `T` | Length of training data | Affects the quality and rank properties of the Hankel matrix |
| `K` | Total simulation length | Determines the overall simulation duration |
| `L` | Prediction horizon | Influences prediction accuracy and control performance |
| `N` | Control horizon | Affects tracking performance |
| `Q` | Output weighting matrix | Tunes output tracking performance |
| `R` | Input weighting matrix | Tunes control effort and smoothness |
| `tau` | Discretization step size | Used for continuous-time system discretization |

---

## ⚙️ Requirements

- MATLAB R2022a or later  
- CVX Toolbox
- SLICOT Basic Systems and Control Toolbox
- MOSEK solver (recommended)  
- Control System Toolbox  
- For water network case:
  - EPANET MATLAB Toolkit (`readEPANETFile`)
  - EPANET input file (e.g., `Net3.inp`)


