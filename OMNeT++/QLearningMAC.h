// =============================================================================
//  QLearningMAC.h
//  Self-Adaptive Contention Window for WSN Using Q-Learning
//  OMNeT++ 6.0 / Castalia 3.3 — IEEE 802.15.4 MAC header
//
//  Thesis    : Self-Adaptive Contention Window for WSN Using Q-Learning
//  University : Ammar Telidji University of Laghouat
//  Authors   : Makhloufi Mohammed Abderrahim — Makhloufi Khadidja
//  Supervisor : Dr. Lakhdar Oulad Djedid
//  Year      : 2025/2026
//
//  Parameters consistent with Chapter 4 (Table 4.2):
//    alpha = 0.1 | gamma = 0.9 | epsilon_0 = 1.0 | epsilon_min = 0.01
//    lambda = 0.005 | CW_min = 8 | CW_max = 256
//    Actions = {8, 16, 32, 64, 256} | States = 27 | Q-entries = 135
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
//  IEEE 802.15.4 constants — consistent with Chapter 4 Table 4.2
// ─────────────────────────────────────────────────────────────────────────────
static constexpr int    CW_MIN_DEF    = 8;
static constexpr int    CW_MAX_DEF    = 256;
static constexpr int    N_ACTIONS     = 5;
static constexpr int    N_STATES      = 27;             // 3^3
static const     int    ACTIONS[5]    = {8,16,32,64,256};

// Timing (IEEE 802.15.4 standard)
static constexpr double SLOT_DUR      = 0.000320;       // 320 µs per slot
static constexpr double TX_DUR        = 0.004;          // 4 ms TX window
static constexpr double ACK_TIMEOUT   = 0.001;          // 1 ms ACK timeout

// Energy model
static constexpr double E_TX_OK       = 0.001;          // J — successful TX
static constexpr double E_TX_FAIL     = 0.0015;         // J — collision

// ─────────────────────────────────────────────────────────────────────────────
//  Reward function constants — Equations (4.5)(4.6)(4.7) Chapter 4
// ─────────────────────────────────────────────────────────────────────────────
static constexpr double R_SUCCESS     = +10.0;
static constexpr double R_COLLISION   = -10.0;
static constexpr double W1            =  0.5;   // success/collision weight
static constexpr double W2            =  0.3;   // energy weight
static constexpr double W3            =  0.2;   // delay weight
static constexpr double BETA          =  0.1;   // energy penalty coefficient
static constexpr double ALPHA_D       =  1.0;   // delay penalty coefficient

// ─────────────────────────────────────────────────────────────────────────────
//  State discretization thresholds — Table 4.1 Chapter 4
// ─────────────────────────────────────────────────────────────────────────────
static constexpr double COLL_LO  = 0.20;   // collision rate  low/med boundary
static constexpr double COLL_HI  = 0.50;   // collision rate  med/high boundary
static constexpr double BUSY_LO  = 0.30;   // channel busy    low/med boundary
static constexpr double BUSY_HI  = 0.70;   // channel busy    med/high boundary
static constexpr double QUEUE_LO = 0.33;   // queue occ.      empty/half boundary
static constexpr double QUEUE_HI = 0.66;   // queue occ.      half/full boundary


// =============================================================================
//  MACState — discrete 3-component state  s_t = (delta_t, rho_t, q_t)
//  Equation (4.1) in Chapter 4
// =============================================================================
struct MACState {
    int delta;   // collision rate level  : 0=low  1=med  2=high
    int rho;     // channel busy level    : 0=low  1=med  2=high
    int queue;   // queue occupancy level : 0=empty 1=half 2=full

    /// Map 3-component tuple to flat Q-table row index [0, 26]
    int toIndex() const {
        return delta * 9 + rho * 3 + queue;
    }
};


// =============================================================================
//  MACStats — sliding window network statistics
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
//  QLearningMAC — OMNeT++ simple module
// =============================================================================
class QLearningMAC : public cSimpleModule
{
  private:
    // ── Parameters (read from omnetpp.ini / NED) ──────────────────────────
    int    nodeId;
    double txPowerDbm;
    double alpha;           // learning rate         (default 0.1)
    double gamma_;          // discount factor       (default 0.9)
    double epsilon;         // current ε             (starts at 1.0)
    double epsilonMin;      // minimum ε             (default 0.01)
    double lambdaDecay;     // decay rate λ          (default 0.005)
    int    cwMin;           // CW_min                (default 8)
    int    cwMax;           // CW_max                (default 256)
    int    macMaxRetries;   // max retransmissions   (default 3)

    // ── Q-table: 27 states × 5 actions = 135 entries ─────────────────────
    //  Memory: 135 × 4 bytes (float32) = 540 bytes — well within 4 KB SRAM
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

    // ── OMNeT++ self-messages ─────────────────────────────────────────────
    cMessage *backoffTimer;
    cMessage *txTimer;
    cMessage *ackTimer;

    // ── Output vectors ────────────────────────────────────────────────────
    cOutVector vecReward;
    cOutVector vecCW;
    cOutVector vecCollRate;
    cOutVector vecEpsilon;
    cOutVector vecPDR;
    cOutVector vecEnergy;

    // ── Cumulative tracking ───────────────────────────────────────────────
    double totalEnergy;
    int    totalSuccess;
    int    totalPackets;

  protected:
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

    // ── Q-Learning core — Algorithm 4.1 Chapter 4 ─────────────────────────
    int    selectAction(const MACState &s);
    void   updateQTable(const MACState &s, int aIdx,
                        double reward, const MACState &ns);
    double computeReward(bool success, int cw, double energy);
    void   decayEpsilon();

    // ── Channel & backoff ─────────────────────────────────────────────────
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
