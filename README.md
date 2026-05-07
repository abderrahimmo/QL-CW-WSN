# QL-CW-WSN: Q-Learning Contention Window Adaptation for IEEE 802.15.4 WSNs

This repository contains the Python implementation and simulation code
developed for the master's thesis:

> Self-Adaptive Contention Window for Wireless Sensor Networks Using Q-Learning

The goal is to evaluate a lightweight, fully distributed Q-Learning mechanism
(QL-CW) for contention window (CW) adaptation in IEEE 802.15.4-like
CSMA/CA wireless sensor networks, and to compare it with a Binary Exponential
Backoff (BEB) baseline.

## Repository Structure

- `src/qlcw_sim.py`  
  Main Python script implementing:
  - the tabular Q-Learning agent (QL-CW),
  - a BEB-based MAC baseline,
  - a discrete-time WSN simulator (nodes, channel, traffic generator),
  - and the logic to run a heavy-load scenario for different network sizes.

- `results/heavy_load_N20_100.txt`  
  Example output file produced by `qlcw_sim.py` for a heavy-load scenario
  with:
  - packet-generation rate `lambda = 1.5 pkt/s` per node,
  - simulation time `T = 60 s`,
  - network sizes `N = {20, 40, 60, 80, 100}`.

These results correspond to the curves and tables presented in Chapter 4
of the thesis.

## Requirements

The code is written in pure Python and only uses the standard library
(`random`, `math`). On Ubuntu, you can install Python 3 using:

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv
