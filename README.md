# Self-Adaptive Contention Window for WSN Using Q-Learning

> **Master's Thesis** — University of Ammar Telidji, Laghouat, Algeria  
> **Authors:** Makhloufi Mohammed Abderrahim — Makhloufi Khadidja  
> **Supervisor:** Dr. Lakhdar Oulad Djedid  
> **Year:** 2025/2026  
> **Specialization:** Networks, Systems and Distributed Applications

---

## Description

This repository contains the complete source code for the thesis:

**"Self-Adaptive Contention Window for Wireless Sensor Networks Using Q-Learning"**

The proposed **QL-CW** mechanism replaces the standard Binary Exponential Backoff (BEB) of IEEE 802.15.4 with a Q-Learning agent that autonomously selects the optimal Contention Window (CW) based on locally observed network conditions — without any centralized coordination.

---

## Key Results (Chapter 4)

| Metric | BEB | MILD | **QL-CW (Proposed)** |
|--------|-----|------|----------------------|
| Collision Rate (100 nodes) | 42.3 % | 29.5 % | **17.1 %** (↓59 % vs BEB) |
| Network Lifetime | baseline | +8 % | **+18 %** |
| PDR (heavy traffic) | 65.8 % | 72.6 % | **86.5 %** |
| Q-Table Memory | — | — | **< 270 bytes** |
| Convergence | — | — | **150–200 episodes** |

---

## How It Works

Each sensor node is an independent Q-Learning agent:

```
Observe state s_t = (δ_t, ρ_t, q_t)
    │
    ▼
Select CW ∈ {8, 16, 32, 64, 256}   ← ε-greedy policy
    │
    ▼
Execute CSMA/CA backoff with selected CW
    │
    ▼
Observe outcome (success / collision)
    │
    ▼
Compute reward: r = w1·r_success + w2·r_energy + w3·r_delay
    │
    ▼
Update Q-table: Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') − Q(s,a)]
    │
    ▼
Decay ε:  ε = max(ε_min, ε_0 · e^(−λt))
```

### State Space — 27 States (3³)

| Component | Symbol | Levels |
|-----------|--------|--------|
| Collision Rate | δ_t | Low [0, 0.2) / Med [0.2, 0.5) / High [0.5, 1] |
| Channel Busy | ρ_t | Low [0, 0.3) / Med [0.3, 0.7) / High [0.7, 1] |
| Queue Occupancy | q_t | Empty [0, 33%) / Half [33, 66%) / Full [66, 100%] |

### Action Space — 5 Actions

```
A = {8, 16, 32, 64, 256}   →   Q-table: 27 × 5 = 135 entries (< 270 bytes)
```

### Hyperparameters

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Learning rate | α | 0.1 |
| Discount factor | γ | 0.9 |
| Initial exploration | ε₀ | 1.0 |
| Min exploration | ε_min | 0.01 |
| Decay rate | λ | 0.005 |

---

## Repository Structure

```
QL-CW-WSN/
│
├── README.md
│
├── Python/
│   └── q_learning_wsn.py        Python implementation
│                                 Agent + WSN Simulator + Evaluation + Plots
│
└── OMNeT++/
    ├── QLearningMAC.h           C++ class declaration
    ├── QLearningMAC.cc          C++ full implementation
    ├── QLearningMAC.ned         NED module description
    ├── QLearningMAC.msg         Message definitions
    └── omnetpp.ini              Simulation scenarios
```

---

## Run — Python

### Requirements
```bash
pip install numpy matplotlib
```

### Run training and evaluation
```bash
cd Python
python q_learning_wsn.py
```

### Output files

| File | Description |
|------|-------------|
| `fig_convergence.png` | Q-Learning convergence (4 subplots) |
| `fig_q_table.png` | Learned Q-Table heatmap |
| `fig_energy_time.png` | Energy vs time — BEB vs QL-CW |
| `fig_energy_nodes.png` | Energy vs number of nodes |
| `fig_collision_nodes.png` | Collision rate vs nodes |
| `fig_pdr_traffic.png` | PDR vs traffic load |
| `q_table_trained.json` | Saved Q-table for deployment |

---

## Run — OMNeT++ / Castalia

### Requirements
- OMNeT++ 6.0 → https://omnetpp.org/download/
- Castalia 3.3 → https://github.com/boulis/Castalia

### Setup
```bash
# 1. Compile message definitions
cd OMNeT++
opp_msgc QLearningMAC.msg

# 2. Copy module into Castalia
cp QLearningMAC.h   Castalia/src/node/communication/mac/
cp QLearningMAC.cc  Castalia/src/node/communication/mac/
cp QLearningMAC.ned Castalia/src/node/communication/mac/

# 3. Rebuild
cd Castalia/Castalia
./makemake && make

# 4. Run a scenario
opp_run -f omnetpp.ini -c DensitySweep_QL
```

### Available scenarios in `omnetpp.ini`

| Config | Description |
|--------|-------------|
| `Scenario_BEB_Light` | BEB baseline — 20 nodes, light traffic |
| `DensitySweep_QL` | QL-CW density sweep (20–100 nodes) |
| `DensitySweep_BEB` | BEB density sweep for comparison |
| `TrafficSweep_QL` | QL-CW traffic sweep (60 nodes) |
| `TrafficSweep_BEB` | BEB traffic sweep for comparison |
| `HyperSensitivity_Alpha` | α sensitivity analysis |
| `HyperSensitivity_Gamma` | γ sensitivity analysis |

---

## References

- Watkins & Dayan (1992) — Q-Learning
- IEEE Std 802.15.4-2020 — Low-Rate Wireless Networks
- Sutton & Barto (2018) — Reinforcement Learning: An Introduction
- Liu et al. (2022) — Deep RL for WSN MAC
- Zhang et al. (2023) — Q-Learning for energy-efficient WSN MAC
- Wang et al. (2024) — Multi-agent DQN for contention window

---

## License

Academic thesis — University of Ammar Telidji Laghouat.  
© 2025/2026 Makhloufi Mohammed Abderrahim — Makhloufi Khadidja.26 Makhloufi Mohammed Abderrahim — Makhloufi Khadidja. All rights reserved.
