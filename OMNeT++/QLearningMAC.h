// =============================================================================
//  QLearningMAC.h
//  Self-Adaptive Contention Window for WSN Using Q-Learning
//  OMNeT++ 6.0 / Castalia 3.3 — IEEE 802.15.4 MAC extension
//
//  Thesis : Self-Adaptive Contention Window for WSN Using Q-Learning
//  University : Ammar Telidji University of Laghouat
//  Authors   : Makhloufi Mohammed Abderrahim — Makhloufi Khadidja
//  Supervisor : Dr. Lakhdar Oulad Djedid
//  Year      : 2025/2026
// =============================================================================

#ifndef QLEARNING_MAC_H
#define QLEARNING_MAC_H

#include <omnetpp.h>
#include <cmath>
#include <random>
#include <algorithm>
using namespace omnetpp;

// ─── IEEE 802.15.4 constants ──────────────────────────────────
static constexpr int    CW_MIN     = 8;
static constexpr int    CW_MAX     = 256;
static constexpr int    N_ACTIONS  = 5;
static constexpr int    N_STATES   = 27;
static const     int    ACTIONS[5] = {8, 16, 32, 64, 256};

// ─── Q-Learning hyperparameters ───────────────────────────────
static constexpr double ALPHA       = 0.1;
static constexpr double GAMMA       = 0.9;
static constexpr double EPS_0       = 1.0;
static constexpr double EPS_MIN     = 0.01;
static constexpr double LAM         = 0.005;

// ─── Reward weights ───────────────────────────────────────────
static constexpr double R_SUCC      = +10.0;
static constexpr double R_COLL      = -10.0;
static constexpr double W1          =  0.5;
static constexpr double W2          =  0.3;
static constexpr double W3          =  0.2;
static constexpr double BETA        =  0.1;


// =============================================================================
//  State — discrete 3-component tuple (delta, rho, queue)
// =============================================================================
struct MACState {
    int delta;  // collision rate : 0=low  1=med  2=high
    int rho;    // channel busy   : 0=low  1=med  2=high
    int queue;  // queue occ.     : 0=empty 1=half 2=full

    int toIndex() const { return delta*9 + rho*3 + queue; }
};


// =============================================================================
//  Q-Table — 27 x 5 float matrix
// =============================================================================
class QTable {
public:
    float data[N_STATES][N_ACTIONS];

    QTable() {
        for (int s = 0; s < N_STATES; ++s)
            for (int a = 0; a < N_ACTIONS; ++a)
                data[s][a] = 0.0f;
    }

    int bestAction(int s) const {
        int b = 0;
        for (int a = 1; a < N_ACTIONS; ++a)
            if (data[s][a] > data[s][b]) b = a;
        return b;
    }

    float maxQ(int s) const {
        float m = data[s][0];
        for (int a = 1; a < N_ACTIONS; ++a)
            if (data[s][a] > m) m = data[s][a];
        return m;
    }

    void update(int s, int a, float r, int sn) {
        float td = r + (float)GAMMA * maxQ(sn) - data[s][a];
        data[s][a] += (float)ALPHA * td;
    }
};


// =============================================================================
//  Statistics tracker
// =============================================================================
struct MACStats {
    int total_tx   = 0;
    int total_coll = 0;
    int busy_count = 0;
    int queue_len  = 0;
    int queue_max  = 32;

    double collRate()  const { return total_tx>0 ? (double)total_coll/total_tx : 0; }
    double busyRatio() const { return total_tx>0 ? (double)busy_count/total_tx : 0; }
    double queueOcc()  const { return (double)queue_len / queue_max; }

    void record(bool coll, bool busy) {
        total_tx++;
        if (coll) total_coll++;
        if (busy) busy_count++;
    }
};


// =============================================================================
//  Discretization helpers
// =============================================================================
static int disc(double v, double lo, double hi) {
    return (v < lo) ? 0 : (v < hi) ? 1 : 2;
}

static MACState buildState(const MACStats& s) {
    return { disc(s.collRate(),  0.20, 0.50),
             disc(s.busyRatio(), 0.30, 0.70),
             disc(s.queueOcc(),  0.33, 0.66) };
}


// =============================================================================
//  QLearningMAC — OMNeT++ simple module
// =============================================================================
class QLearningMAC : public cSimpleModule
{
  private:
    QTable    qtable;
    MACStats  stats;
    MACState  curState;
    int       curActionIdx = 0;
    int       curCW        = CW_MIN;
    double    epsilon      = EPS_0;
    int       episode      = 0;

    std::mt19937 rng;

    cMessage* backoffTimer = nullptr;
    cMessage* txTimer      = nullptr;

    int    nodeId;
    double txPowerDbm;

    cOutVector vecReward, vecCW, vecCollRate, vecEpsilon;

  protected:
    void initialize() override;
    void handleMessage(cMessage* msg) override;
    void finish() override;

  private:
    int    selectAction(const MACState& s);
    double computeReward(bool success, int cw, double energy);
    void   learnFromOutcome(bool success, bool busy);
    void   decayEpsilon();
    void   startBackoff(int cw);
    void   attemptTx();
    bool   simCollision(int cw);
    void   logMetrics(double reward);
};


// ─── initialize ───────────────────────────────────────────────
void QLearningMAC::initialize() {
    nodeId     = par("nodeId");
    txPowerDbm = par("txPowerDbm").doubleValue();
    rng.seed(nodeId + 42);
    curState = buildState(stats);

    vecReward.setName("reward");
    vecCW.setName("selected_CW");
    vecCollRate.setName("collision_rate");
    vecEpsilon.setName("epsilon");

    startBackoff(CW_MIN);
    EV << "[QLearningMAC] Node " << nodeId << " initialized.\n";
}

// ─── handleMessage ────────────────────────────────────────────
void QLearningMAC::handleMessage(cMessage* msg) {
    if (msg == backoffTimer) {
        attemptTx();
    } else if (msg == txTimer) {
        bool coll = simCollision(curCW);
        bool busy = coll || (std::uniform_real_distribution<>(0,1)(rng) < 0.4);
        learnFromOutcome(!coll, busy);
        curState     = buildState(stats);
        curActionIdx = selectAction(curState);
        curCW        = ACTIONS[curActionIdx];
        startBackoff(curCW);
    } else {
        delete msg;
    }
}

// ─── selectAction ─────────────────────────────────────────────
int QLearningMAC::selectAction(const MACState& s) {
    std::uniform_real_distribution<> d(0, 1);
    if (d(rng) < epsilon) {
        std::uniform_int_distribution<> a(0, N_ACTIONS-1);
        return a(rng);
    }
    return qtable.bestAction(s.toIndex());
}

// ─── computeReward ────────────────────────────────────────────
double QLearningMAC::computeReward(bool success, int cw, double energy) {
    double rs = success ? R_SUCC : R_COLL;
    double re = -BETA * energy;
    double rd = -(double)cw / CW_MAX;
    return W1*rs + W2*re + W3*rd;
}

// ─── learnFromOutcome ─────────────────────────────────────────
void QLearningMAC::learnFromOutcome(bool success, bool busy) {
    double energy  = 0.001 * (success ? 1.0 : 1.5);
    stats.record(!success, busy);
    if (stats.queue_len > 0 && success) stats.queue_len--;
    else stats.queue_len = std::min(stats.queue_max, stats.queue_len + 1);

    double reward = computeReward(success, curCW, energy);
    MACState ns   = buildState(stats);
    qtable.update(curState.toIndex(), curActionIdx,
                  (float)reward, ns.toIndex());
    decayEpsilon();
    logMetrics(reward);
}

// ─── decayEpsilon ─────────────────────────────────────────────
void QLearningMAC::decayEpsilon() {
    episode++;
    epsilon = std::max(EPS_MIN, EPS_0 * std::exp(-LAM * episode));
}

// ─── startBackoff ─────────────────────────────────────────────
void QLearningMAC::startBackoff(int cw) {
    std::uniform_int_distribution<> d(0, cw-1);
    double delay = d(rng) * 0.000320;   // 320 µs per slot
    if (backoffTimer) cancelAndDelete(backoffTimer);
    backoffTimer = new cMessage("backoff");
    scheduleAt(simTime() + delay, backoffTimer);
}

// ─── attemptTx ────────────────────────────────────────────────
void QLearningMAC::attemptTx() {
    if (txTimer) cancelAndDelete(txTimer);
    txTimer = new cMessage("tx");
    scheduleAt(simTime() + 0.004, txTimer);
}

// ─── simCollision ─────────────────────────────────────────────
bool QLearningMAC::simCollision(int cw) {
    double p = 1.0 - std::pow(1.0 - 1.0 / std::max(1, cw), 3);
    return std::uniform_real_distribution<>(0,1)(rng) < p;
}

// ─── logMetrics ───────────────────────────────────────────────
void QLearningMAC::logMetrics(double reward) {
    vecReward.record(reward);
    vecCW.record(curCW);
    vecCollRate.record(stats.collRate());
    vecEpsilon.record(epsilon);
}

// ─── finish ───────────────────────────────────────────────────
void QLearningMAC::finish() {
    EV << "[Node " << nodeId << "]\n"
       << "  total_tx   = " << stats.total_tx    << "\n"
       << "  collisions = " << stats.total_coll  << "\n"
       << "  coll_rate  = " << stats.collRate()  << "\n"
       << "  epsilon    = " << epsilon            << "\n"
       << "  episodes   = " << episode            << "\n";
    if (backoffTimer) cancelAndDelete(backoffTimer);
    if (txTimer)      cancelAndDelete(txTimer);
}

Define_Module(QLearningMAC);

#endif // QLEARNING_MAC_H
