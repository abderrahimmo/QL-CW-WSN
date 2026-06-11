```markdown
# QL-CW-WSN: Q-Learning Contention Window Adaptation for IEEE 802.15.4 WSNs

This repository contains the Python implementation and simulation code
developed for the master's thesis:

> Self-Adaptive Contention Window for Wireless Sensor Networks
> Using Q-Learning

The goal is to evaluate a lightweight and fully distributed Q-Learning
mechanism, called **QL-CW**, for contention window (CW) adaptation in
IEEE 802.15.4-like CSMA/CA wireless sensor networks, and to compare it
with a Binary Exponential Backoff (**BEB**) baseline.

---

## Repository Structure

```text
QL-CW-WSN/
├── src/
│   ├── qlcw_sim.py
│   └── plot_results.py
├── data/
│   ├── results.csv
│   ├── fig_collisions.png
│   ├── fig_pdr.png
│   ├── fig_delay.png
│   └── fig_energy.png
└── README.md
```

- `src/qlcw_sim.py`  
  Main Python script implementing the QL-CW mechanism (Q-table of
  189 entries, action set {8,16,32,64,128,256,512}), the BEB baseline,
  the shared channel model, the traffic generator, and the statistics
  collection module.

- `src/plot_results.py`  
  Python script used to generate the performance figures from
  `data/results.csv`.

- `data/results.csv`  
  CSV file containing the numerical simulation results.

- `data/fig_collisions.png`  
  Figure showing total collisions versus number of nodes.

- `data/fig_pdr.png`  
  Figure showing Packet Delivery Ratio versus number of nodes.

- `data/fig_delay.png`  
  Figure showing average end-to-end delay versus number of nodes.

- `data/fig_energy.png`  
  Figure showing average energy consumption per node versus number
  of nodes.

---

## Requirements

The simulator uses Python 3. The main simulation script relies on the
Python standard library (`random`, `math`, `csv`, `collections`),
while the plotting script requires `matplotlib`.

On Ubuntu, the required packages can be installed using:

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv python3-matplotlib
```

If needed, `matplotlib` can also be installed using:

```bash
pip3 install matplotlib
```

---

## Simulation Scenario

The final configuration used in the thesis is:

| Parameter              | Value                                 |
|------------------------|---------------------------------------|
| Packet generation rate | `lambda = 5.0 pkt/s` per node         |
| Simulation time        | `T = 200 s`                           |
| Number of runs         | `5`                                   |
| Network sizes          | `N = {20, 40, 60, 80, 100}`           |
| BEB CW range           | `CWmin = 8`, `CWmax = 128`            |
| QL-CW action set       | `{8, 16, 32, 64, 128, 256, 512}`      |
| Q-table size           | `27 states × 7 actions = 189 entries` |
| Memory footprint       | `~378 bytes (16-bit fixed-point)`     |
| Learning rate          | `alpha = 0.1`                         |
| Discount factor        | `gamma = 0.9`                         |
| Initial exploration    | `epsilon = 1.0`                       |
| Minimum exploration    | `epsilon_min = 0.01`                  |
| Exploration decay      | `0.001`                               |
| Reward weights         | `(w1=1.0, w2=0.05, w3=0.05)`          |

> **Reward weights:** `w1` controls the success/collision term (dominant),
> `w2` penalises energy expenditure, and `w3` penalises large contention
> windows. The dominant weight on `w1` reflects the design priority of
> collision avoidance.

---

## Running the Simulator

From the root directory of the repository, run:

```bash
python3 src/qlcw_sim.py
```

The simulator evaluates two schemes:

- **BEB**: Binary Exponential Backoff baseline
  (`CWmin=8`, `CWmax=128`).
- **QL-CW**: proposed Q-Learning based contention window adaptation
  mechanism (action set `{8,16,32,64,128,256,512}`, 189 Q-table entries).

After execution, the numerical results are saved in:

```text
results.csv
```

---

## Generating the Figures

To generate the performance figures from the CSV results, run:

```bash
python3 src/plot_results.py
```

The generated figures are saved in the `data/` directory:

```text
data/fig_collisions.png
data/fig_pdr.png
data/fig_delay.png
data/fig_energy.png
```

These figures correspond to the performance curves presented in
Chapter 4 of the thesis.

---

## Main Results

The following table summarises the averaged results obtained under the
heavy-load scenario (`lambda = 5.0 pkt/s`, `T = 200 s`, 5 runs):

| Nodes | BEB PDR (%) | QL-CW PDR (%) | BEB Collisions | QL-CW Collisions | BEB Delay (s) | QL-CW Delay (s) | BEB Energy | QL-CW Energy |
|------:|------------:|--------------:|---------------:|-----------------:|--------------:|----------------:|-----------:|-------------:|
|    20 |      100.00 |         99.93 |          2,337 |            2,222 |       0.00699 |         0.18693 |    1117.18 |      1106.66 |
|    40 |       99.99 |         99.89 |         12,426 |           11,501 |       0.01051 |         0.24268 |    1308.44 |      1283.30 |
|    60 |       99.99 |         99.78 |         40,801 |           37,204 |       0.02279 |         0.45740 |    1677.89 |      1615.74 |
|    80 |       71.25 |         92.11 |        322,617 |          140,302 |      11.16495 |         5.13978 |    4743.89 |      2673.79 |
|   100 |       49.74 |         66.85 |        382,666 |          235,046 |      17.54930 |        11.79191 |    4323.88 |      3018.93 |

### Key Observations

- **Non-saturated regime (N ≤ 60):** Both schemes maintain PDR above
  99.7%. QL-CW reduces collisions by ~5–9% and per-node energy by up
  to ~4%, at the cost of a sub-second delay increase due to exploration
  over the extended action set.

- **Saturated regime (N ≥ 80):** QL-CW outperforms BEB on every metric:
  - Collisions reduced by **~56.5%** at N=80 and **~38.6%** at N=100
  - PDR improved by **+20.9 pp** at N=80 and **+17.1 pp** at N=100
  - Average delay reduced by **~54%** at N=80 and **~32.8%** at N=100
  - Per-node energy reduced by **~43.6%** at N=80 and **~30.2%** at N=100

---

## Reproducibility

To reproduce the results and figures:

```bash
git clone https://github.com/abderrahimmo/QL-CW-WSN.git
cd QL-CW-WSN
python3 src/qlcw_sim.py
python3 src/plot_results.py
```

After execution, the following files are generated or updated:

```text
results.csv
data/fig_collisions.png
data/fig_pdr.png
data/fig_delay.png
data/fig_energy.png
```

---

## Citation

If you use this code or results in your work, please cite:

> [abderrahim MAKHLOUFI], *Self-Adaptive Contention Window for Wireless Sensor
> Networks Using Q-Learning*, Master's Thesis, Université Amar Telidji
> de Laghouat, 2025/2026.
```
