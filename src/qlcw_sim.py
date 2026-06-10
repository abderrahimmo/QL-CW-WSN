#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# QL-CW vs BEB simulator for IEEE 802.15.4-like CSMA/CA MAC
import random, math, csv
from collections import deque

class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9,
                 eps0=1.0, eps_min=0.01, decay=0.001):
        self.n_states=n_states; self.n_actions=n_actions
        self.alpha=alpha; self.gamma=gamma
        self.eps=eps0; self.eps_min=eps_min; self.decay=decay; self.t=0
        self.Q=[[0.0]*n_actions for _ in range(n_states)]
    def select_action(self, s):
        if random.random()<self.eps: return random.randrange(self.n_actions)
        row=self.Q[s]; best=row[0]; best_a=0
        for a in range(1,self.n_actions):
            if row[a]>best: best=row[a]; best_a=a
        return best_a
    def update(self, s, a, r, s_next):
        td=(r+self.gamma*max(self.Q[s_next]))-self.Q[s][a]
        self.Q[s][a]+=self.alpha*td
        self.t+=1
        self.eps=max(self.eps_min, self.eps*math.exp(-self.decay))

class BaseNode:
    def __init__(self, node_id, queue_max=50, window=50):
        self.id=node_id; self.queue_max=queue_max; self.window=window
        self.state="IDLE"; self.backoff_counter=0; self.busy_until=-1
        self.queue=deque()
        self.attempts=0; self.successes=0; self.collisions=0
        self.total_delay=0.0; self.packets_delivered=0; self.dropped=0; self.energy=0.0
    def generate_packet(self, t_now):
        if len(self.queue)<self.queue_max: self.queue.append(t_now)
        else: self.dropped+=1
    def pick_cw(self): raise NotImplementedError
    def decide_transmit(self, channel_busy):
        if self.state=="IDLE":
            if not self.queue: return False
            cw=self.pick_cw()
            self.backoff_counter=random.randint(0,cw-1)
            self.state="BACKOFF"; return False
        if self.state=="BACKOFF":
            if channel_busy: return False
            if self.backoff_counter>0: self.backoff_counter-=1; return False
            return True
        return False
    def on_transmission_result(self, success, finish_time): raise NotImplementedError

class NodeQL(BaseNode):
    def __init__(self, node_id, agent, cw_values, channel,
                 queue_max=50, window=50, weights=(1.0,0.05,0.05)):
        super().__init__(node_id, queue_max, window)
        self.agent=agent; self.cw_values=cw_values; self.max_cw=max(cw_values)
        self.channel=channel; self.weights=weights
        self.col_window=deque(maxlen=window); self.cur_s=0; self.cur_a=0
    def compute_state(self):
        delta=sum(self.col_window)/len(self.col_window) if self.col_window else 0.0
        rho=self.channel.busy_ratio()
        q_ratio=len(self.queue)/self.queue_max
        i_d=0 if delta<0.2 else (1 if delta<0.5 else 2)
        i_r=0 if rho<0.3 else (1 if rho<0.7 else 2)
        i_q=0 if q_ratio<0.33 else (1 if q_ratio<0.66 else 2)
        return i_d*9+i_r*3+i_q
    def pick_cw(self):
        self.cur_s=self.compute_state()
        self.cur_a=self.agent.select_action(self.cur_s)
        return self.cw_values[self.cur_a]
    def on_transmission_result(self, success, finish_time):
        self.attempts+=1
        if success:
            self.successes+=1; self.col_window.append(0)
            if self.queue:
                arrival=self.queue.popleft()
                self.total_delay+=(finish_time-arrival); self.packets_delivered+=1
            r_success=+1.0
        else:
            self.collisions+=1; self.col_window.append(1); r_success=-1.0
        E_tx=1.0; self.energy+=E_tx
        cw_used=self.cw_values[self.cur_a]
        r_delay=-(cw_used/self.max_cw); r_energy=-E_tx
        w1,w2,w3=self.weights
        r=w1*r_success+w2*r_energy+w3*r_delay
        self.agent.update(self.cur_s, self.cur_a, r, self.compute_state())
        self.state="IDLE"

class NodeBEB(BaseNode):
    def __init__(self, node_id, cw_min, cw_max, queue_max=50, window=50):
        super().__init__(node_id, queue_max, window)
        self.cw_min=cw_min; self.cw_max=cw_max; self.cw_current=cw_min
    def pick_cw(self): return self.cw_current
    def on_transmission_result(self, success, finish_time):
        self.attempts+=1
        if success:
            self.successes+=1; self.cw_current=self.cw_min
            if self.queue:
                arrival=self.queue.popleft()
                self.total_delay+=(finish_time-arrival); self.packets_delivered+=1
        else:
            self.collisions+=1; self.cw_current=min(self.cw_current*2, self.cw_max)
        self.energy+=1.0; self.state="IDLE"

class Channel:
    def __init__(self, busy_window=50):
        self.busy_until=-1; self.busy_hist=deque(maxlen=busy_window)
    def is_busy(self, t): return t<self.busy_until
    def record(self, b): self.busy_hist.append(1 if b else 0)
    def busy_ratio(self):
        return sum(self.busy_hist)/len(self.busy_hist) if self.busy_hist else 0.0

def run_simulation(nodes, channel, n_slots, slot_duration, lam, seed):
    random.seed(seed); p_arr=lam*slot_duration; total_generated=0; tx_slots=TX_SLOTS
    for t in range(n_slots):
        t_now=t*slot_duration
        for node in nodes:
            if node.state=="TX" and t>=node.busy_until: node.state="IDLE"
        channel_busy=channel.is_busy(t); channel.record(channel_busy)
        for node in nodes:
            if random.random()<p_arr:
                node.generate_packet(t_now); total_generated+=1
        tx_requests=[]
        if not channel_busy:
            for node in nodes:
                if node.state=="TX": continue
                if node.decide_transmit(channel_busy): tx_requests.append(node)
        else:
            for node in nodes:
                if node.state!="TX": node.decide_transmit(channel_busy)
        if tx_requests:
            end=t+tx_slots; channel.busy_until=end; finish_time=end*slot_duration
            success=(len(tx_requests)==1)
            for node in tx_requests:
                node.busy_until=end; node.state="TX"
                node.on_transmission_result(success, finish_time)
    return total_generated

def compute_metrics(nodes, total_generated, n_nodes):
    td=sum(n.total_delay for n in nodes); deliv=sum(n.packets_delivered for n in nodes)
    return {
        "generated":total_generated,
        "collisions":sum(n.collisions for n in nodes),
        "delivered":deliv,
        "pdr":deliv/total_generated if total_generated else 0.0,
        "avg_delay":td/deliv if deliv else 0.0,
        "avg_energy_per_node":sum(n.energy for n in nodes)/n_nodes if n_nodes else 0.0,
    }

def build_nodes(algo, n_nodes, channel):
    nodes=[]
    if algo=="BEB":
        for i in range(n_nodes): nodes.append(NodeBEB(i,CW_MIN,CW_MAX,QUEUE_MAX,WINDOW))
    else:
        for i in range(n_nodes):
            agent=QLearningAgent(27,len(CW_VALUES),**Q_PARAMS)
            nodes.append(NodeQL(i,agent,CW_VALUES,channel,QUEUE_MAX,WINDOW,REWARD_WEIGHTS))
    return nodes

def run_averaged(algo, n_nodes, n_slots):
    acc=None
    for run in range(NUM_RUNS):
        channel=Channel(busy_window=WINDOW); nodes=build_nodes(algo,n_nodes,channel)
        gen=run_simulation(nodes,channel,n_slots,SLOT_DURATION,LAMBDA,seed=run)
        m=compute_metrics(nodes,gen,n_nodes)
        if acc is None: acc={k:0.0 for k in m}
        for k in m: acc[k]+=m[k]
    for k in acc: acc[k]/=NUM_RUNS
    return acc

# ===== Configuration =====
SLOT_DURATION=0.001
SIM_TIME=60.0
LAMBDA=10.0
TX_SLOTS=1
CW_MIN=8; CW_MAX=128                      # BEB: standard IEEE 802.15.4 bounds
CW_VALUES=[8,16,32,64,128,256,512]        # QL-CW: extended action set (contribution)
QUEUE_MAX=50; WINDOW=50; NUM_RUNS=5
Q_PARAMS={"alpha":0.1,"gamma":0.9,"eps0":1.0,"eps_min":0.01,"decay":0.001}
REWARD_WEIGHTS=(1.0,0.05,0.05)            # (success, energy, delay)
NODE_COUNTS=[20,40,60,80,100]

def main():
    n_slots=int(SIM_TIME/SLOT_DURATION)
    print("=== heavy load (lambda={:.1f} pkt/s), T={:.0f}s, runs={} ==="
          .format(LAMBDA,SIM_TIME,NUM_RUNS))
    rows=[]
    for n in NODE_COUNTS:
        beb=run_averaged("BEB",n,n_slots); ql=run_averaged("QL-CW",n,n_slots)
        print(f"\n--- N={n} nodes ---")
        print("BEB  : PDR={:6.2f}%  Coll={:8.0f}  Delay={:.5f}s  En={:.2f}".format(
            beb["pdr"]*100,beb["collisions"],beb["avg_delay"],beb["avg_energy_per_node"]))
        print("QL-CW: PDR={:6.2f}%  Coll={:8.0f}  Delay={:.5f}s  En={:.2f}".format(
            ql["pdr"]*100,ql["collisions"],ql["avg_delay"],ql["avg_energy_per_node"]))
        rows.append({"nodes":n,"beb_pdr":beb["pdr"]*100,"ql_pdr":ql["pdr"]*100,
            "beb_collisions":beb["collisions"],"ql_collisions":ql["collisions"],
            "beb_delay":beb["avg_delay"],"ql_delay":ql["avg_delay"],
            "beb_energy":beb["avg_energy_per_node"],"ql_energy":ql["avg_energy_per_node"]})
    with open("results.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("\n[OK] results.csv written")

if __name__=="__main__":
    main()
