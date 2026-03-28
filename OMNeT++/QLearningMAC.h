// =============================================================================
//  QLearningMAC.h
//  Self-Adaptive Contention Window for WSN Using Q-Learning
//  OMNeT++ 6.0 / Castalia 3.3 — IEEE 802.15.4 MAC header
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
#include <string>

using namespace omnetpp;

// ─────────────────────────────────────────────────────────────────────────────
//  IEEE 802.15.4 MAC constants
// ─────────────────────────────────────────────────────────────────────────────
static constexpr int    N_ACTIONS     = 5;
static constexpr int    N_STATES      = 27;        // 3^3
static const     int    ACTIONS[5]    = {8, 16, 32, 64, 256};

// Timing constants (seconds)
static constexpr double SLOT_DURATION = 0.000320;  // 320 µs per IEEE 802.15.4
static constexpr double TX_DURATION   = 0.004;     // 4 ms transmission window
static constexpr double ACK_TIMEOUT   = 0.001;     // 1 ms ACK timeout

// Energy model (Joules)
static constexpr double E_TX          = 0.001;     // energy per transmission

// ─────────────────────────────────────────────────────────────────────────────
//  Reward function constants
// ─────────────────────────────────────────────────────────────────────────────
static constexpr double R_SUCCESS     = +10.0;
static constexpr double R_COLLISION   = -10.0;
static constexpr double W1            =  0.5;
static constexpr double W2            =  0.3;
static constexpr double W3            =  0.2;
static constexpr double BETA          =  0.1;
static constexpr double ALPHA_D       =  1.0;


// =============================================================================
//  MACState — discrete 3-component state tuple
// =============================================================================
struct MACState {
    int delta;   // collision rate  : 0=low  1=med  2=high
    int rho;     // channel busy    : 0=low  1=med  2=high
    int queue;   // queue occupancy : 0=empty 1=half 2=full

    int toIndex() const {
        return delta * 9 + rho * 3 + queue;
    }
};


// =============================================================================
//  MACStats — sliding window statistics tracker
// =============================================================================
struct MACStats {
    int total_tx   = 0;
    int total_coll = 0;
    int busy_count = 0;
    int queue_len  = 0;
    int queue_max  = 32;

    double collRate()  const {
        return (total_tx > 0) ? (double)total_coll / total_tx : 0.0;
    }
    double busyRatio() const {
        return (total_tx > 0) ? (double)busy_count / total_tx : 0.0;
    }
    double queueOcc()  const {
        return (double)queue_len / std::max(1, queue_max);
    }

    void recordTx(bool collision, bool busy) {
        total_tx++;
        if (collision) total_coll++;
        if (busy)      busy_count++;
    }
};


// =============================================================================
//  QLearningMAC — OMNeT++ simple module declaration
// =============================================================================
class QLearningMAC : public cSimpleModule
{
  private:
    // ── Q-Learning parameters (read from omnetpp.ini) ─────────────────────
    int    nodeId;
    double txPowerDbm;
    double alpha;        // learning rate
    double gamma_;       // discount factor (gamma_ avoids conflict with std)
    double epsilon;      // current exploration rate
    double epsilonMin;   // minimum exploration rate
    double lambdaDecay;  // epsilon decay rate
    int    cwMin;
    int    cwMax;
    int    macMaxRetries;

    // ── Q-table [27 states × 5 actions] ──────────────────────────────────
    float  qTable[N_STATES][N_ACTIONS];

    // ── Current state and action ──────────────────────────────────────────
    MACState curState;
    int      curActionIdx;
    int      curCW;
    int      episode;
    int      retryCount;

    // ── Network statistics ────────────────────────────────────────────────
    MACStats stats;

    // ── Random number generator ───────────────────────────────────────────
    std::mt19937 rng;

    // ── OMNeT++ self-messages (timers) ────────────────────────────────────
    cMessage *backoffTimer;
    cMessage *txTimer;
    cMessage *ackTimer;

    // ── Output vectors for result collection ──────────────────────────────
    cOutVector vecReward;
    cOutVector vecCW;
    cOutVector vecCollRate;
    cOutVector vecEpsilon;

  protected:
    // ── OMNeT++ lifecycle ─────────────────────────────────────────────────
    virtual void initialize(int stage = 0) override;
    virtual void handleMessage(cMessage *msg) override;
    virtual void finish() override;

  private:
    // ── MAC event handlers ────────────────────────────────────────────────
    void attemptTransmission();
    void handleCollision();
    void handleSuccess();
    void handleUpperPacket(cMessage *msg);
    void handleRadioFrame(cMessage *msg);

    // ── Q-Learning core ───────────────────────────────────────────────────
    int    selectAction(const MACState &s);
    void   updateQTable(const MACState &s,
                        int actionIdx,
                        double reward,
                        const MACState &ns);
    double computeReward(bool success, int cw, double energy);
    void   decayEpsilon();

    // ── Channel & backoff helpers ─────────────────────────────────────────
    void   scheduleBackoff(int cw);
    bool   senseChannel();
    void   updateQueueLen(bool success);

    // ── State helpers ─────────────────────────────────────────────────────
    static int      disc(double v, double lo, double hi);
    static MACState buildState(const MACStats &s);

    // ── Logging ───────────────────────────────────────────────────────────
    void logMetrics(double reward);
};

#endif // QLEARNING_MAC_H
