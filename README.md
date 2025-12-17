# RL-Hopper: Vertical Jumping Agent with Deep Reinforcement Learning

This repository contains the source code for the BSc thesis titled **"Robustness analyses of a mechanical system controlled by reinforcement learning"**, written by Katalin Juhász.
The project implements a Deep Q-Network (DQN) agent capable of controlling a custom 1-DOF vertical jumping robot (Hopper) to reach a target height.

The simulation environment uses the `gymnasium` interface, and the agent is trained using PyTorch with Experience Replay and $\epsilon$-greedy exploration.

## Repository Structure

The codebase is organized into modular Python scripts:

* **`Custom_Environment.py`**: Defines the `MassActuatorEnv` class (physics, rewards, state transitions).
* **`Neural_Network.py`**: Implements the DQN architecture (3-layer MLP).
* **`Memory.py`**: Implements the Replay Memory buffer.
* **`Training.py`**: Main script for training the agents.
* **`Testing.py`**: Script for evaluating trained agents and generating safety maps.
* **`Monitoring.py`**: Utilities for data visualization and logging.
* **`Device.py`**: Helper for hardware acceleration (CUDA/MPS/CPU).

## Installation & Requirements

The project is implemented in **Python 3**. To run the simulations, install the required dependencies:

```bash
pip install numpy matplotlib torch gymnasium tensorboard
```

The code does not generate the directory structure automatically, so the following structure needs to be manually created in the project root:

```text
project_root/
├── agents/
├── figures/
│   ├── basins/
│   ├── last_ep/
│   └── test_ep/
└── logs/
    ├── basins/
    ├── all_ep/
    └── test_ep/
```
