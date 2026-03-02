# QL-CW-WSN
Self-Adaptive Contention Window for WSN Using Q-Learning
# Self-Adaptive Contention Window for WSN Using Q-Learning

> **Master's Thesis** — University of Ammar Telidji, Laghouat, Algeria  
> **Authors:** Makhloufi Mohammed Abderrahim — Makhloufi Khadidja  
> **Supervisor:** Dr. Lakhdar Oulad Djedid  
> **Academic Year:** 2025/2026  
> **Specialization:** Distributed Networks, Systems, and Applications

---

## 📋 Description

This repository contains the complete source code for the thesis:

**"Self-Adaptive Contention Window for WSN Using Q-Learning"**

The proposed **QL-CW** framework replaces the standard Binary Exponential Backoff (BEB) algorithm of IEEE 802.15.4 with a Q-Learning agent that dynamically selects the optimal Contention Window (CW) size based on locally observed network conditions.

---

## 🧠 How It Works

Each sensor node acts as an independent Q-Learning agent that:

1. **Observes** the current network state: `(collision_rate, channel_busy, queue_occupancy)`
2. **Selects** a CW value from `{8, 16, 32, 64, 256}` using an ε-greedy policy
3. **Receives** a multi-objective reward: `r = w1·r_success + w2·r_energy + w3·r_delay`
4. **Updates** its Q-table using the Bellman equation
5. **Repeats** — converging to an optimal policy within ~150 episodes

---

## 📁 Repository Structure

```
QL-CW-WSN/
│
├── README.md                    ← This file
│
├── Python/
│   └── q_learning_wsn.py        ← Q-Learning agent + WSN simulator + evaluation
│
└── OMNeT++/
    ├── QLearningMAC.h           ← C++ MAC module for OMNeT++/Castalia
    ├── QLearningMAC.ned         ← NED network description file
    └── omnetpp.ini              ← Simulation scenarios configuration
```

---

## ▶️ How to Run — Python

### Requirements
```bash
pip install numpy matplotlib
```

### Run training and evaluation
```bash
cd Python
python q_learning_wsn.py
```

### Output files generated
| File | Description |
|------|-------------|
| `fig_convergence.png` | Q-Learning convergence curves (4 subplots) |
| `fig_q_table.png` | Learned Q-Table heatmap |
| `fig_comparison.png` | BEB vs MILD vs QL-CW comparison |
| `q_table_trained.json` | Saved Q-table for deployment |

---

## ▶️ How to Run — OMNeT++/Castalia

### Requirements
- OMNeT++ 6.0 → https://omnetpp.org/download/
- Castalia 3.3 → https://github.com/boulis/Castalia

### Setup
```bash
# 1. Copy module files into Castalia
cp OMNeT++/QLearningMAC.h   Castalia/src/node/communication/mac/
cp OMNeT++/QLearningMAC.ned Castalia/src/node/communication/mac/

# 2. Rebuild Castalia
cd Castalia/Castalia
./makemake && make

# 3. Run a scenario
cd Simulations/QL_WSN
opp_run -f omnetpp.ini -c DensitySweep_QL
```

---

## 📊 Key Results

| Metric | BEB | MILD | **QL-CW (Proposed)** |
|--------|-----|------|----------------------|
| Collision Rate (100 nodes) | 42.3% | 29.5% | **17.1%** |
| Network Lifetime | baseline | +8% | **+18%** |
| PDR (heavy traffic) | 65.8% | 72.6% | **86.5%** |
| Q-Table Memory | — | — | **< 270 bytes** |
| Convergence | — | — | **~150 episodes** |

---

## ⚙️ Q-Learning Parameters

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Learning rate | α | 0.1 |
| Discount factor | γ | 0.9 |
| Initial exploration | ε₀ | 1.0 |
| Min exploration | ε_min | 0.01 |
| Decay rate | λ | 0.005 |
| State space size | \|S\| | 27 |
| Action space size | \|A\| | 5 |
| Q-Table entries | \|S\|×\|A\| | 135 |

---

## 📚 References

- Watkins & Dayan (1992) — Q-Learning algorithm
- IEEE Std 802.15.4-2020 — Low-Rate Wireless Networks standard
- Liu et al. (2022) — DRL-based MAC for WSN collision mitigation
- Zhang et al. (2023) — Q-Learning for energy-efficient WSN MAC
- Sutton & Barto (2018) — Reinforcement Learning: An Introduction

---

## 📄 License

This project is submitted as an academic thesis at the University of Ammar Telidji Laghouat.  
© 2025/2026 Makhloufi Mohammed Abderrahim — Makhloufi Khadidja. All rights reserved.
