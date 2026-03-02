"""
=============================================================================
  q_learning_wsn.py
  Self-Adaptive Contention Window for WSN Using Q-Learning
  Python Implementation — Training, Evaluation and Visualization

  Thesis : Self-Adaptive Contention Window for WSN Using Q-Learning
  University : Ammar Telidji University of Laghouat
  Authors   : Makhloufi Mohammed Abderrahim — Makhloufi Khadidja
  Supervisor : Dr. Lakhdar Oulad Djedid
  Year      : 2025/2026
=============================================================================
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import json
from collections import deque

# ─────────────────────────────────────────────────────────────
# IEEE 802.15.4 parameters
# ─────────────────────────────────────────────────────────────
CW_MIN  = 8
CW_MAX  = 256
ACTIONS = [CW_MIN, 16, 32, 64, CW_MAX]   # |A| = 5

# State discretization thresholds
COLL_TH  = [0.2, 0.5]    # collision rate  : low / medium / high
BUSY_TH  = [0.3, 0.7]    # channel busy    : low / medium / high
QUEUE_TH = [0.33, 0.66]  # queue occupancy : empty / half / full

# Q-Learning hyperparameters
ALPHA       = 0.1
GAMMA       = 0.9
EPSILON_0   = 1.0
EPSILON_MIN = 0.01
LAMBDA      = 0.005

# Reward weights
R_SUCCESS   =  10.0
R_COLLISION = -10.0
W1, W2, W3  =  0.5, 0.3, 0.2
BETA        =  0.1
ALPHA_D     =  1.0

# Simulation
N_EPISODES      = 500
E_TX_PER_PACKET = 0.001   # Joules per attempt


# ─────────────────────────────────────────────────────────────
# Helper : state discretization
# ─────────────────────────────────────────────────────────────
def discretize(value, thresholds):
    for i, t in enumerate(thresholds):
        if value < t:
            return i
    return len(thresholds)


def get_state(collision_rate, channel_busy, queue_occ):
    return (discretize(collision_rate, COLL_TH),
            discretize(channel_busy,   BUSY_TH),
            discretize(queue_occ,      QUEUE_TH))


def state_to_index(state):
    d, r, q = state
    return d * 9 + r * 3 + q   # 27 states total


# ─────────────────────────────────────────────────────────────
# Reward function
# ─────────────────────────────────────────────────────────────
def compute_reward(success, cw_selected, energy):
    r_s = R_SUCCESS if success else R_COLLISION
    r_e = -BETA * energy
    r_d = -ALPHA_D * (cw_selected / CW_MAX)
    return W1 * r_s + W2 * r_e + W3 * r_d


# ─────────────────────────────────────────────────────────────
# Q-Learning Agent
# ─────────────────────────────────────────────────────────────
class QLearningAgent:
    def __init__(self, node_id=0):
        self.node_id = node_id
        self.q_table = np.zeros((27, len(ACTIONS)), dtype=np.float32)
        self.epsilon  = EPSILON_0
        self.episode  = 0
        self.rewards_log        = []
        self.epsilon_log        = []
        self.cw_log             = []
        self.collision_rate_log = []

    def select_action(self, state):
        s = state_to_index(state)
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        return ACTIONS[int(np.argmax(self.q_table[s]))]

    def update(self, state, action_cw, reward, next_state):
        s  = state_to_index(state)
        ns = state_to_index(next_state)
        a  = ACTIONS.index(action_cw)
        td = reward + GAMMA * float(np.max(self.q_table[ns])) \
             - self.q_table[s, a]
        self.q_table[s, a] += ALPHA * td

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN,
                           EPSILON_0 * np.exp(-LAMBDA * self.episode))
        self.episode += 1

    def save(self, path="q_table_trained.json"):
        with open(path, "w") as f:
            json.dump({"q_table": self.q_table.tolist(),
                       "epsilon": float(self.epsilon),
                       "episode": self.episode}, f, indent=2)
        print(f"[Node {self.node_id}] Q-table saved → '{path}'")

    def load(self, path="q_table_trained.json"):
        with open(path) as f:
            d = json.load(f)
        self.q_table = np.array(d["q_table"], dtype=np.float32)
        self.epsilon  = d["epsilon"]
        self.episode  = d["episode"]


# ─────────────────────────────────────────────────────────────
# WSN Channel Simulator
# ─────────────────────────────────────────────────────────────
class WSNChannelSimulator:
    RATES = {"light": 0.05, "medium": 0.15, "heavy": 0.30}

    def __init__(self, n_nodes=20, traffic="medium"):
        self.n_nodes      = n_nodes
        self.arrival_rate = self.RATES.get(traffic, 0.15)
        self.reset()

    def reset(self):
        self.queue      = 0
        self.time       = 0
        self.total_tx   = 0
        self.total_coll = 0
        self.busy_slots = 0

    def step(self, cw):
        self.queue = min(self.queue + np.random.poisson(self.arrival_rate), 32)
        n_cont  = max(1, self.n_nodes // 4)
        p_coll  = min(1.0 - (1.0 - 1.0 / max(1, cw)) ** (n_cont - 1), 0.95)
        collision = random.random() < p_coll
        success   = (not collision) and self.queue > 0
        if success:
            self.queue = max(0, self.queue - 1)
        busy = collision or (random.random() < 0.4)
        self.total_tx   += 1
        self.total_coll += int(collision)
        self.busy_slots += int(busy)
        self.time       += 1
        return {
            "success":         success,
            "collision":       collision,
            "collision_rate":  self.total_coll / max(1, self.total_tx),
            "channel_busy":    self.busy_slots / max(1, self.time),
            "queue_occupancy": self.queue / 32.0,
            "energy":          E_TX_PER_PACKET * (1 + 0.5 * (not success)),
        }


# ─────────────────────────────────────────────────────────────
# BEB Baseline
# ─────────────────────────────────────────────────────────────
class BEBBaseline:
    def __init__(self):
        self.cw = CW_MIN

    def select_cw(self, success):
        self.cw = CW_MIN if success else min(self.cw * 2, CW_MAX)
        return self.cw


# ─────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────
def train_agent(n_nodes=60, traffic="medium",
                n_episodes=N_EPISODES, verbose=True):
    agent   = QLearningAgent()
    channel = WSNChannelSimulator(n_nodes, traffic)
    window  = deque(maxlen=50)
    cum     = []

    for ep in range(n_episodes):
        channel.reset()
        state   = get_state(0.0, 0.1, 0.2)
        cw      = agent.select_action(state)
        out     = channel.step(cw)
        reward  = compute_reward(out["success"], cw, out["energy"])
        nstate  = get_state(out["collision_rate"],
                            out["channel_busy"],
                            out["queue_occupancy"])
        agent.update(state, cw, reward, nstate)
        agent.decay_epsilon()
        window.append(reward)
        cum.append(float(np.mean(window)))
        agent.rewards_log.append(reward)
        agent.epsilon_log.append(agent.epsilon)
        agent.cw_log.append(cw)
        agent.collision_rate_log.append(out["collision_rate"])
        if verbose and ep % 100 == 0:
            print(f"  Ep {ep:4d} | CW={cw:3d} | R={reward:+6.2f} "
                  f"| ε={agent.epsilon:.4f} | coll={out['collision_rate']:.3f}")

    print(f"\n[Done] {n_episodes} episodes — final ε={agent.epsilon:.4f}")
    return agent, cum


# ─────────────────────────────────────────────────────────────
# Evaluation : QL-CW vs BEB vs MILD
# ─────────────────────────────────────────────────────────────
def evaluate(n_nodes_list=None, traffic="medium", steps=200):
    if n_nodes_list is None:
        n_nodes_list = [20, 40, 60, 80, 100]
    res = {a: {"collision": [], "energy": [], "pdr": []}
           for a in ["QL-CW", "BEB", "MILD"]}

    for n in n_nodes_list:
        print(f"\n--- {n} nodes ---")
        agent, _ = train_agent(n, traffic, n_episodes=300, verbose=False)

        for algo in ["BEB", "MILD", "QL-CW"]:
            ch = WSNChannelSimulator(n, traffic)
            beb  = BEBBaseline()
            mild_cw = CW_MIN
            state   = get_state(0.0, 0.1, 0.2)
            coll_list, eng_list, succ_list = [], [], []

            for _ in range(steps):
                if algo == "BEB":
                    cw = beb.select_cw(succ_list[-1] if succ_list else True)
                elif algo == "MILD":
                    if succ_list and succ_list[-1]:
                        mild_cw = max(CW_MIN, mild_cw - 1)
                    elif succ_list:
                        mild_cw = min(CW_MAX, int(mild_cw * 1.5))
                    cw = mild_cw
                else:
                    cw = agent.select_action(state)

                out = ch.step(cw)
                if algo == "QL-CW":
                    state = get_state(out["collision_rate"],
                                      out["channel_busy"],
                                      out["queue_occupancy"])
                coll_list.append(out["collision_rate"])
                eng_list.append(out["energy"])
                succ_list.append(1 if out["success"] else 0)

            res[algo]["collision"].append(float(np.mean(coll_list)))
            res[algo]["energy"].append(float(np.mean(eng_list)) * steps)
            res[algo]["pdr"].append(float(np.mean(succ_list)) * 100)
            print(f"  {algo:<6} → coll={res[algo]['collision'][-1]:.3f} "
                  f"PDR={res[algo]['pdr'][-1]:.1f}%")

    res["n_nodes"] = n_nodes_list
    return res


# ─────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────
def plot_convergence(agent, cum, path="fig_convergence.png"):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Q-Learning Convergence — Self-Adaptive CW for WSN",
                 fontsize=13, fontweight="bold")
    axes[0, 0].plot(cum, color="navy")
    axes[0, 0].set_title("(a) Average Reward")
    axes[0, 0].set_xlabel("Episode"); axes[0, 0].set_ylabel("Mean Reward")
    axes[0, 0].grid(True, ls="--", alpha=0.5)

    axes[0, 1].plot(agent.epsilon_log, color="darkorange")
    axes[0, 1].set_title("(b) Exploration Rate ε")
    axes[0, 1].set_xlabel("Episode"); axes[0, 1].set_ylabel("ε")
    axes[0, 1].grid(True, ls="--", alpha=0.5)

    axes[1, 0].plot(agent.cw_log, color="green", alpha=0.7, lw=0.8)
    axes[1, 0].set_title("(c) Selected CW")
    axes[1, 0].set_xlabel("Episode"); axes[1, 0].set_ylabel("CW value")
    axes[1, 0].set_yticks(ACTIONS)
    axes[1, 0].grid(True, ls="--", alpha=0.5)

    axes[1, 1].plot(agent.collision_rate_log, color="crimson", alpha=0.7, lw=0.8)
    axes[1, 1].set_title("(d) Collision Rate")
    axes[1, 1].set_xlabel("Episode"); axes[1, 1].set_ylabel("Collision Rate")
    axes[1, 1].grid(True, ls="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved → '{path}'")
    plt.show()


def plot_comparison(res, path="fig_comparison.png"):
    n   = res["n_nodes"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Performance Comparison : BEB vs MILD vs QL-CW",
                 fontsize=13, fontweight="bold")
    cfg = [("collision", "Collision Rate"),
           ("energy",    "Total Energy (J)"),
           ("pdr",       "PDR (%)")]
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    for ax, (key, title) in zip(axes, cfg):
        for algo, col in zip(["BEB", "MILD", "QL-CW"], colors):
            ax.plot(n, res[algo][key], marker="o", color=col,
                    lw=2, label=algo)
        ax.set_title(title); ax.set_xlabel("Number of Nodes")
        ax.legend(); ax.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved → '{path}'")
    plt.show()


def plot_qtable(agent, path="fig_q_table.png"):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(agent.q_table, aspect="auto", cmap="RdYlGn")
    plt.colorbar(im, ax=ax, label="Q-value")
    ax.set_xticks(range(len(ACTIONS)))
    ax.set_xticklabels([f"CW={a}" for a in ACTIONS])
    ax.set_xlabel("Action (CW value)")
    ax.set_ylabel("State index (0–26)")
    ax.set_title("Learned Q-Table — Self-Adaptive CW for WSN")
    labels = [f"δ={d},ρ={r},q={q}"
              for d in "LMH" for r in "LMH" for q in "EHF"]
    ax.set_yticks(range(27)); ax.set_yticklabels(labels, fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved → '{path}'")
    plt.show()


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Self-Adaptive CW for WSN — Q-Learning")
    print("  University of Ammar Telidji — Laghouat")
    print("=" * 60)

    print("\n[1] Training (60 nodes, medium traffic) ...")
    agent, cum = train_agent(n_nodes=60, traffic="medium",
                              n_episodes=N_EPISODES, verbose=True)
    agent.save("q_table_trained.json")

    print("\n[2] Convergence plots ...")
    plot_convergence(agent, cum, "fig_convergence.png")

    print("\n[3] Q-table heatmap ...")
    plot_qtable(agent, "fig_q_table.png")

    print("\n[4] Comparative evaluation ...")
    results = evaluate([20, 40, 60, 80, 100], "medium", 200)

    print("\n[5] Comparison plots ...")
    plot_comparison(results, "fig_comparison.png")

    print("\n" + "=" * 60)
    print(f"{'Nodes':>6} | {'BEB Coll':>10} | {'MILD Coll':>10} | "
          f"{'QL Coll':>10} | {'BEB PDR':>8} | {'QL PDR':>8}")
    print("-" * 60)
    for i, n in enumerate(results["n_nodes"]):
        print(f"{n:6d} | "
              f"{results['BEB']['collision'][i]:10.4f} | "
              f"{results['MILD']['collision'][i]:10.4f} | "
              f"{results['QL-CW']['collision'][i]:10.4f} | "
              f"{results['BEB']['pdr'][i]:8.1f}% | "
              f"{results['QL-CW']['pdr'][i]:8.1f}%")
    print("=" * 60)
