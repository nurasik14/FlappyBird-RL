# FlappyBird-RL

A simple Deep Q-Network (DQN) implementation for Flappy Bird used for learning Reinforcement Learning (RL) concepts and DQN applications.

> Note: This repository follows and reproduces the code and lessons from Jonny Code's tutorial series. It is a learning / follow-along project and not an entirely original build — credit to Jonny Code for the tutorial and core ideas.

Table of Contents
- [Project status](#project-status)
- [What this repository is](#what-this-repository-is)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Play (manual)](#play-manual)
  - [Train (DQN)](#train-dqn)
  - [Evaluate / Visualize](#evaluate--visualize)
- [Configuration and hyperparameters](#configuration-and-hyperparameters)
- [Files and structure (expected)](#files-and-structure-expected)
- [Results and notes](#results-and-notes)
- [Contributing and issues](#contributing-and-issues)
- [License](#license)
- [Acknowledgements & references](#acknowledgements--references)

## Project status
Learning project — works as a tutorial reproduction. Not production-ready. Expect to tune hyperparameters and possibly adapt file names depending on any local changes.

## What this repository is
This repo implements a DQN agent for Flappy Bird following Jonny Code's tutorial series. The implementation and experiments are intended as an educational exercise to learn RL basics and DQN mechanics (replay buffer, target network, epsilon-greedy, etc.). Any deviations or experiments are my own adaptations while following the tutorial.

## Features
- DQN agent for Flappy Bird
- Experience replay buffer
- Target network updates
- Training loop and simple evaluation/play mode
- Configurable hyperparameters for experimentation

## Requirements
- Python 3.8+
- Typical dependencies:
  - numpy
  - torch (PyTorch)
  - pygame (for the game environment / rendering)
  - matplotlib (optional, for plotting training curves)
- A requirements.txt is recommended (example below).

Example requirements.txt entries:
numpy
torch
pygame
matplotlib

## Installation
1. Clone the repository:
   git clone https://github.com/nurasik14/FlappyBird-RL.git
   cd FlappyBird-RL

2. Create and activate a virtual environment (recommended):
   python -m venv venv
   source venv/bin/activate  # macOS / Linux
   venv\Scripts\activate     # Windows

3. Install dependencies:
   pip install -r requirements.txt
   (or) pip install numpy torch pygame matplotlib

## Usage

Note: exact script names may differ if you modified filenames. Below are common commands used in tutorial-style repos.

Play (manual)
- Run a script that starts the game and allows manual play or visualizes an agent:
  python play.py

Train (DQN)
- Run the training script:
  python train.py
- Common options (if implemented): --episodes, --batch-size, --lr, --gamma, --save-path

Evaluate / Visualize
- After training, run evaluation or rendering:
  python evaluate.py --model-path models/dqn_final.pth

If your repository uses different filenames, adjust commands accordingly.

## Configuration and hyperparameters
Typical DQN config options to tune:
- learning rate (lr)
- discount factor (gamma)
- batch size
- replay buffer size
- epsilon start/end and decay schedule
- target network update frequency
- number of training episodes / steps

Store these in a config dict or CLI flags to make experiments reproducible.

## Files and structure (expected)
This section lists common files you may have or add. Replace names to match your repo if different.
- README.md — this file
- requirements.txt — Python package list
- train.py — training loop
- play.py / evaluate.py — play or evaluate the trained agent
- models.py — network architecture (DQN)
- replay_buffer.py — experience replay buffer
- utils.py — helper functions (preprocessing, logging)
- assets/ or screenshots/ — optional visuals and saved plots
- models/ — saved checkpoints

## Results and notes
- Training RL agents can be noisy; results will vary by random seed and hyperparameters.
- Use smaller training runs to verify end-to-end functionality before long experiments.
- Save periodic checkpoints and training logs (rewards per episode) for later analysis.

## Contributing and issues
This repo is primarily a learning reproduction. If you want to suggest improvements or corrections:
- Open an issue describing the change or bug.
- I welcome PRs that improve documentation, add reproducible experiments, or fix bugs.

## License
Choose and include a license file (e.g., MIT). If you want, add a LICENSE file to the repo.

## Acknowledgements & references
- This repository follows the tutorial series by Jonny Code — all credit for the original walkthrough and teaching goes to Jonny Code.
- Useful RL references:
  - "Human-level control through deep reinforcement learning" — Mnih et al., 2015 (DQN)
  - Official PyTorch docs: https://pytorch.org
