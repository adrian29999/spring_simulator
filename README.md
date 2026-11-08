# spring_simulator

An interactive mass-spring-damper simulator with an optional PID controller, built with NumPy and Matplotlib. Adjust the physical parameters and PID gains with sliders and watch the response, phase portrait and numerical-error diagnostics update live.

## Features

- Mass-spring-damper model: `m·x'' + c·x' + k·x = F`
- Optional PID controller (Kp, Ki, Kd)
- RK4 time integration
- Live plots: position and velocity vs time, phase portrait, RK4 error vs exact solution, and formula-vs-scan fast-zeta comparison
- Two settling-time markers: a fixed 0.1% band and an adjustable "band 2"
- **Crit** and **fast ζ** snap buttons for `m`, `k`, `c`, `Kp` and `Kd`
- Info panel with the equation of motion, effective constants, state-space matrices, eigenvalues and damping classification

## Requirements

- Python 3.9+
- NumPy
- Matplotlib

## Installation

```bash
git clone https://github.com/adrian29999/spring_simulator.git
cd spring_simulator
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip3 install -r requirements.txt
```

## Usage

```bash
python3 spring_sim.py
```

