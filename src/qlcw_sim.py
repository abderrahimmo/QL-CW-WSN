#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# QL-CW vs BEB simulator for IEEE 802.15.4-like MAC
# Single file, only uses Python standard libraries.

import random
import math


# =========================
# Q-Learning Agent (for QL-CW)
# =========================

class QLearningAgent:
    def __init__(self,
                 n_states: int,
                 n_actions: int,
                 alpha: float = 0.1,
                 gamma: float = 0.9,
                 eps0: float = 1.0,
                 eps_min: float = 0.01,
                 decay: float = 0.001):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps0
        self.eps_min = eps_min
        self.decay = decay
        self.t = 0  # number of decisions
        # Q-table as list of lists: Q[s][a]
        self.Q = [[0.0 for _ in range(n_actions)] for _ in range(n_states)]

    def select_action(self, state_idx: int) -> int:
        """Return action index using epsilon-greedy policy."""
        # Exploration
        if random.random() < self.eps:
            return random.randrange(self.n_actions)
        # Exploitation
        row = self.Q[state_idx]
        max_val = max(row)
        # argmax (first index of max value)
        for a, v in enumerate(row):
            if v == max_val:
                return a
        return 0  # fallback

    def update(self, s: int, a: int, r: float, s_next: int):
        """Q-learning update rule."""
        max_next = max(self.Q[s_next])
        td_target = r + self.gamma * max_next
        td_error = td_target - self.Q[s][a]
        self.Q[s][a] += self.alpha * td_error

        # update epsilon (exponential decay)
        self.t += 1
        self.eps = max(self.eps_min, self.eps * math.exp(-self.decay))


# =========================
# Node with QL-CW (learning)
# =========================

class NodeQL:
    def __init__(self, node_id: int,
                 agent: QLearningAgent,
                 cw_values: list,
                 queue_max: int = 50):
        self.id = node_id
        self.agent = agent
        self.cw_values = cw_values  # list of CW values (ints)
        self.queue_max = queue_max

        # MAC state
        self.state = "IDLE"
        self.backoff_counter = 0
        self.current_state_idx = 0
        self.current_action_idx = 0

        # Queue: store arrival times (for delay)
        self.queue = []

        # Statistics
        self.attempts = 0
        self.successes = 0
        self.collisions = 0
        self.total_delay = 0.0
        self.packets_delivered = 0
        self.energy = 0.0  # simple energy metric

        # Histories for delta (collision rate) and rho (busy ratio)
        self.collision_history = []  # 1 for collision, 0 for success
        self.attempt_history = []    # 1 per attempt
        self.busy_history = []       # 1 if channel busy, 0 if idle

    # -------- Traffic --------
    def generate_packet(self, current_time: float):
        """Generate one packet (if queue not full)."""
        if len(self.queue) < self.queue_max:
            self.queue.append(current_time)

    # -------- Channel busy record --------
    def record_channel_busy(self, busy_flag: bool):
        """Call once per slot to record whether channel was busy."""
        self.busy_history.append(1 if busy_flag else 0)

    # -------- RL state computation --------
    def compute_state_index(self,
                            window_attempts: int = 50,
                            window_busy: int = 50) -> int:
        """Discretise (delta, rho, q) into state index in [0..26]."""

        # delta: collision rate over last 'window_attempts' attempts
        if len(self.attempt_history) == 0:
            delta = 0.0
        else:
            last_att = self.attempt_history[-window_attempts:]
            last_col = self.collision_history[-window_attempts:]
            total_att = sum(last_att)
            total_col = sum(last_col)
            delta = (total_col / total_att) if total_att > 0 else 0.0

        # rho: busy ratio over last 'window_busy' slots
        if len(self.busy_history) == 0:
            rho = 0.0
        else:
            last_busy = self.busy_history[-window_busy:]
            rho = sum(last_busy) / len(last_busy)

        # q: queue occupancy ratio
        q_ratio = len(self.queue) / self.queue_max

        # bucket functions (same thresholds as in chapter 4)
        def bucket_delta(x):
            if x < 0.2:
                return 0  # low
            elif x < 0.5:
                return 1  # medium
            else:
                return 2  # high

        def bucket_rho(x):
            if x < 0.3:
                return 0
            elif x < 0.7:
                return 1
            else:
                return 2

        def bucket_q(x):
            if x < 0.33:
                return 0  # empty
            elif x < 0.66:
                return 1  # half
            else:
                return 2  # full

        i_d = bucket_delta(delta)
        i_r = bucket_rho(rho)
        i_q = bucket_q(q_ratio)

        # Encode (i_d, i_r, i_q) into single index [0..26]
        state_idx = i_d * 9 + i_r * 3 + i_q
        return state_idx

    # -------- MAC logic: decide whether to transmit this slot --------
    def decide_transmit(self, channel_was_busy: bool) -> bool:
        """
        Called once per slot before channel decision.
        Returns True if this node wants to transmit in this slot.
        """

        # No packet to send
        if self.state == "IDLE":
            if len(self.queue) == 0:
                return False

            # We have at least one packet, start backoff controlled by Q-learning
            s_idx = self.compute_state_index()
            self.current_state_idx = s_idx
            action_idx = self.agent.select_action(s_idx)
            self.current_action_idx = action_idx
            cw = self.cw_values[action_idx]
            # Random backoff in [0, CW-1]
            self.backoff_counter = random.randint(0, cw - 1)
            self.state = "BACKOFF"
            return False

        if self.state == "BACKOFF":
            # If channel was busy in previous slot, freeze backoff
            if channel_was_busy:
                return False
            # Channel idle -> decrease backoff
            if self.backoff_counter > 0:
                self.backoff_counter -= 1
                return False
            else:
                # Backoff finished, transmit in this slot
                self.state = "TX"
                return True

        # If in TX or any other state, do not schedule another TX start
        return False

    # -------- Process transmission result and update Q-learning --------
    def on_transmission_result(self,
                               success: bool,
                               current_time: float):
        """Called by simulator after channel decides success/collision."""
        # Update attempts and collision history
        self.attempts += 1
        self.attempt_history.append(1)

        if success:
            self.successes += 1
            self.collision_history.append(0)
            # Compute delay (if queue not empty)
            if self.queue:
                arrival_time = self.queue.pop(0)
                delay = current_time - arrival_time
                self.total_delay += delay
                self.packets_delivered += 1
            r_success = +1.0
        else:
            self.collisions += 1
            self.collision_history.append(1)
            r_success = -1.0

        # Simple energy model: one TX attempt costs 1 energy unit
        E_tx = 1.0
        self.energy += E_tx

        # Delay-related part (based on CW used)
        cw_used = self.cw_values[self.current_action_idx]
        max_cw = max(self.cw_values)
        r_delay = - cw_used / max_cw

        # Energy penalty
        r_energy = - E_tx

        # Combine components (weights as example)
        w1, w2, w3 = 0.5, 0.3, 0.2
        r = w1 * r_success + w2 * r_energy + w3 * r_delay

        # Next state
        s_next = self.compute_state_index()
        self.agent.update(self.current_state_idx,
                          self.current_action_idx,
                          r,
                          s_next)

        # Return to IDLE (node will start a new backoff for next packet)
        self.state = "IDLE"


# =========================
# Node with BEB (baseline, no learning)
# =========================

class NodeBEB:
    def __init__(self, node_id: int,
                 cw_min: int,
                 cw_max: int,
                 queue_max: int = 50):
        self.id = node_id
        self.cw_min = cw_min
        self.cw_max = cw_max
        self.cw_current = cw_min
        self.queue_max = queue_max

        self.state = "IDLE"
        self.backoff_counter = 0

        self.queue = []

        # Statistics (same fields as NodeQL for easy comparison)
        self.attempts = 0
        self.successes = 0
        self.collisions = 0
        self.total_delay = 0.0
        self.packets_delivered = 0
        self.energy = 0.0

        self.busy_history = []

    def generate_packet(self, current_time: float):
        if len(self.queue) < self.queue_max:
            self.queue.append(current_time)

    def record_channel_busy(self, busy_flag: bool):
        self.busy_history.append(1 if busy_flag else 0)

    def decide_transmit(self, channel_was_busy: bool) -> bool:
        if self.state == "IDLE":
            if len(self.queue) == 0:
                return False
            # start backoff with current CW
            cw = self.cw_current
            self.backoff_counter = random.randint(0, cw - 1)
            self.state = "BACKOFF"
            return False

        if self.state == "BACKOFF":
            if channel_was_busy:
                return False
            if self.backoff_counter > 0:
                self.backoff_counter -= 1
                return False
            else:
                self.state = "TX"
                return True

        return False

    def on_transmission_result(self,
                               success: bool,
                               current_time: float):
        self.attempts += 1

        if success:
            self.successes += 1
            # reset CW to minimum
            self.cw_current = self.cw_min
            if self.queue:
                arrival_time = self.queue.pop(0)
                delay = current_time - arrival_time
                self.total_delay += delay
                self.packets_delivered += 1
        else:
            self.collisions += 1
            # binary exponential backoff: CW doubles, capped by cw_max
            new_cw = self.cw_current * 2
            if new_cw > self.cw_max:
                new_cw = self.cw_max
            self.cw_current = new_cw

        # Simple energy model
        E_tx = 1.0
        self.energy += E_tx

        self.state = "IDLE"


# =========================
# Channel
# =========================

class Channel:
    def __init__(self):
        self.busy_prev = False

    def decide(self, tx_nodes: list):
        """
        Input: list of Node objects that want to transmit this slot.
        Output: dict {node_id: success_bool}, and busy flag for this slot.
        """
        results = {}
        if len(tx_nodes) == 0:
            busy_now = False
            return results, busy_now

        if len(tx_nodes) == 1:
            busy_now = True
            node = tx_nodes[0]
            results[node.id] = True
        else:
            busy_now = True
            for node in tx_nodes:
                results[node.id] = False

        return results, busy_now


# =========================
# Generic metric computation
# =========================

def compute_metrics(nodes, total_generated, n_nodes):
    total_success = sum(n.successes for n in nodes)
    total_collisions = sum(n.collisions for n in nodes)
    total_delivered = sum(n.packets_delivered for n in nodes)
    total_delay = sum(n.total_delay for n in nodes)
    total_energy = sum(n.energy for n in nodes)

    pdr = (total_delivered / total_generated) if total_generated > 0 else 0.0
    avg_delay = (total_delay / total_delivered) if total_delivered > 0 else 0.0
    avg_energy_per_node = total_energy / n_nodes if n_nodes > 0 else 0.0

    return {
        "generated": total_generated,
        "success": total_success,
        "collisions": total_collisions,
        "delivered": total_delivered,
        "pdr": pdr,
        "avg_delay": avg_delay,
        "avg_energy_per_node": avg_energy_per_node
    }


# =========================
# Simulator for QL-CW
# =========================

class SimulatorQLCW:
    def __init__(self,
                 n_nodes: int,
                 cw_values: list,
                 lambda_arrival: float,
                 slot_duration: float,
                 sim_time: float,
                 q_params: dict):
        self.n_nodes = n_nodes
        self.cw_values = cw_values
        self.lam = lambda_arrival
        self.slot_duration = slot_duration
        self.sim_time = sim_time
        self.n_slots = int(sim_time / slot_duration)

        self.channel = Channel()
        self.nodes = []

        n_states = 27
        n_actions = len(cw_values)

        for i in range(n_nodes):
            agent = QLearningAgent(
                n_states=n_states,
                n_actions=n_actions,
                alpha=q_params["alpha"],
                gamma=q_params["gamma"],
                eps0=q_params["eps0"],
                eps_min=q_params["eps_min"],
                decay=q_params["decay"]
            )
            node = NodeQL(i, agent, cw_values)
            self.nodes.append(node)

        # arrival probability per slot
        self.p_arr = self.lam * self.slot_duration

        self.current_time = 0.0
        self.total_generated = 0

    def run(self):
        for slot in range(self.n_slots):
            self.current_time = slot * self.slot_duration

            # 1) Traffic generation
            for node in self.nodes:
                if random.random() < self.p_arr:
                    node.generate_packet(self.current_time)
                    self.total_generated += 1

            # 2) Decide which nodes attempt to transmit
            tx_nodes = []
            channel_busy_prev = self.channel.busy_prev
            for node in self.nodes:
                wants_tx = node.decide_transmit(channel_busy_prev)
                if wants_tx:
                    tx_nodes.append(node)

            # 3) Channel decision
            results, busy_now = self.channel.decide(tx_nodes)

            # 4) Inform nodes about channel busy/idle this slot
            for node in self.nodes:
                node.record_channel_busy(busy_now)

            # 5) Inform nodes about transmission results (if any)
            for node in self.nodes:
                if node.id in results:
                    success = results[node.id]
                    node.on_transmission_result(success, self.current_time)

            # 6) Update channel busy flag for next slot
            self.channel.busy_prev = busy_now

    def compute_metrics(self):
        return compute_metrics(self.nodes, self.total_generated, self.n_nodes)


# =========================
# Simulator for BEB baseline
# =========================

class SimulatorBEB:
    def __init__(self,
                 n_nodes: int,
                 cw_min: int,
                 cw_max: int,
                 lambda_arrival: float,
                 slot_duration: float,
                 sim_time: float):
        self.n_nodes = n_nodes
        self.cw_min = cw_min
        self.cw_max = cw_max
        self.lam = lambda_arrival
        self.slot_duration = slot_duration
        self.sim_time = sim_time
        self.n_slots = int(sim_time / slot_duration)

        self.channel = Channel()
        self.nodes = []

        for i in range(n_nodes):
            node = NodeBEB(i, cw_min, cw_max)
            self.nodes.append(node)

        self.p_arr = self.lam * self.slot_duration
        self.current_time = 0.0
        self.total_generated = 0

    def run(self):
        for slot in range(self.n_slots):
            self.current_time = slot * self.slot_duration

            # 1) Traffic generation
            for node in self.nodes:
                if random.random() < self.p_arr:
                    node.generate_packet(self.current_time)
                    self.total_generated += 1

            # 2) Decide which nodes attempt to transmit
            tx_nodes = []
            channel_busy_prev = self.channel.busy_prev
            for node in self.nodes:
                wants_tx = node.decide_transmit(channel_busy_prev)
                if wants_tx:
                    tx_nodes.append(node)

            # 3) Channel decision
            results, busy_now = self.channel.decide(tx_nodes)

            # 4) Inform nodes about channel busy/idle this slot
            for node in self.nodes:
                node.record_channel_busy(busy_now)

            # 5) Inform nodes about transmission results
            for node in self.nodes:
                if node.id in results:
                    success = results[node.id]
                    node.on_transmission_result(success, self.current_time)

            # 6) Update channel busy flag
            self.channel.busy_prev = busy_now

    def compute_metrics(self):
        return compute_metrics(self.nodes, self.total_generated, self.n_nodes)


# =========================
# Main: run BEB and QL-CW for comparison
# =========================

if __name__ == "__main__":
    # Sweep over different network sizes for a fixed heavy load
    SLOT_DURATION = 1e-3      # 1 ms per slot
    SIM_TIME = 60.0           # 60 seconds
    LAMBDA = 1.5              # 1.5 pkt/s per node (heavy load)

    CW_MIN = 8
    CW_MAX = 128
    CW_VALUES = [8, 16, 32, 64, 128]

    q_params = {
        "alpha": 0.1,
        "gamma": 0.9,
        "eps0": 1.0,
        "eps_min": 0.01,
        "decay": 0.001
    }

    # قيم N التي نريد تجربتها
    node_counts = [20, 40, 60, 80, 100]

    print("=== Scenario: heavy load (lambda = {:.2f} pkt/s per node), "
          "T = {:.1f} s ===".format(LAMBDA, SIM_TIME))

    for N_NODES in node_counts:
        print("\n------------------------------")
        print(f"Network size: {N_NODES} nodes")
        print("------------------------------")

        # --- Run BEB baseline ---
        random.seed(0)
        sim_beb = SimulatorBEB(n_nodes=N_NODES,
                               cw_min=CW_MIN,
                               cw_max=CW_MAX,
                               lambda_arrival=LAMBDA,
                               slot_duration=SLOT_DURATION,
                               sim_time=SIM_TIME)
        sim_beb.run()
        metrics_beb = sim_beb.compute_metrics()

        # --- Run QL-CW ---
        random.seed(0)
        sim_ql = SimulatorQLCW(n_nodes=N_NODES,
                               cw_values=CW_VALUES,
                               lambda_arrival=LAMBDA,
                               slot_duration=SLOT_DURATION,
                               sim_time=SIM_TIME,
                               q_params=q_params)
        sim_ql.run()
        metrics_ql = sim_ql.compute_metrics()

        # --- Print compact comparison for this N ---
        print("BEB : "
              f"PDR={metrics_beb['pdr']*100:6.2f}%  "
              f"Coll={metrics_beb['collisions']:6d}  "
              f"Delay={metrics_beb['avg_delay']:.5f} s  "
              f"En={metrics_beb['avg_energy_per_node']:.2f}")

        print("QL-CW: "
              f"PDR={metrics_ql['pdr']*100:6.2f}%  "
              f"Coll={metrics_ql['collisions']:6d}  "
              f"Delay={metrics_ql['avg_delay']:.5f} s  "
              f"En={metrics_ql['avg_energy_per_node']:.2f}")
