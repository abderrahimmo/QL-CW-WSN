// =============================================================================
//  QLearningMAC.msg
//  OMNeT++ 6.0 / Castalia 3.3 — Message Definitions
//  Self-Adaptive Contention Window for WSN Using Q-Learning
//
//  Thesis    : Self-Adaptive Contention Window for WSN Using Q-Learning
//  University : Ammar Telidji University of Laghouat
//  Authors   : Makhloufi Mohammed Abderrahim — Makhloufi Khadidja
//  Supervisor : Dr. Lakhdar Oulad Djedid
//  Year      : 2025/2026
//
//  HOW TO COMPILE:
//      opp_msgc QLearningMAC.msg
//  Generates:
//      QLearningMAC_m.h
//      QLearningMAC_m.cc
//  Then include in QLearningMAC.cc:
//      #include "QLearningMAC_m.h"
// =============================================================================

// ── OMNeT++ 6.0 namespace declaration ────────────────────────────────────────
namespace qlcw;


// =============================================================================
//  MacFrame — IEEE 802.15.4 data frame
//  Used for transmissions between sensor nodes
// =============================================================================
packet MacFrame
{
    int    srcAddress   = -1;    // source node ID
    int    dstAddress   = -1;    // destination node ID
    int    seqNumber    = 0;     // frame sequence number
    int    frameType    = 0;     // 0=data  1=ack  2=beacon
    int    payloadBytes = 100;   // payload size in bytes
    int    selectedCW   = 8;     // CW used for this transmission
    int    retryCount   = 0;     // retransmission attempt count
    double rewardValue  = 0.0;   // Q-Learning reward (logging only)
}


// =============================================================================
//  AckFrame — Acknowledgement frame
// =============================================================================
packet AckFrame
{
    int    srcAddress = -1;
    int    dstAddress = -1;
    int    seqNumber  = 0;
}


// =============================================================================
//  BackoffMessage — self-message: backoff timer expired
//  Triggers CCA + transmission attempt
// =============================================================================
message BackoffMessage
{
    int    cwValue      = 8;
    int    backoffSlots = 0;
}


// =============================================================================
//  TxCompleteMessage — self-message: transmission window ended
//  Starts ACK timeout period
// =============================================================================
message TxCompleteMessage
{
    int    seqNumber = 0;
    int    cwValue   = 8;
    double txEnergy  = 0.001;
}


// =============================================================================
//  AckTimeoutMessage — self-message: no ACK received
//  Signals collision to Q-Learning agent
// =============================================================================
message AckTimeoutMessage
{
    int    seqNumber  = 0;
    int    retryCount = 0;
    int    cwValue    = 8;
}


// =============================================================================
//  QLUpdateMessage — self-message: Q-table update trigger
//  Carries full Bellman update data
//  Consistent with Equation (4.8) Chapter 4
// =============================================================================
message QLUpdateMessage
{
    // State s_t = (delta, rho, queue) — Equation (4.1) Chapter 4
    int    stateDelta = 0;    // collision rate  : 0=low 1=med 2=high
    int    stateRho   = 0;    // channel busy    : 0=low 1=med 2=high
    int    stateQueue = 0;    // queue occupancy : 0=empty 1=half 2=full

    // Action a_t
    int    actionIdx  = 0;    // index in {8,16,32,64,256}
    int    cwSelected = 8;

    // Reward r_t — Equation (4.4) Chapter 4
    double reward     = 0.0;
    double energy     = 0.001;

    // Next state s_{t+1}
    int    nextDelta  = 0;
    int    nextRho    = 0;
    int    nextQueue  = 0;

    // Episode metadata
    double epsilon    = 1.0;
    int    episode    = 0;
    int    success    = 0;    // 1=TX success, 0=collision
}
