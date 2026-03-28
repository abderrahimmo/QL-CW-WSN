// =============================================================================
//  QLearningMAC.cc
//  Self-Adaptive Contention Window for WSN Using Q-Learning
//  OMNeT++ 6.0 / Castalia 3.3 — IEEE 802.15.4 MAC implementation
//
//  Thesis : Self-Adaptive Contention Window for WSN Using Q-Learning
//  University : Ammar Telidji University of Laghouat
//  Authors   : Makhloufi Mohammed Abderrahim — Makhloufi Khadidja
//  Supervisor : Dr. Lakhdar Oulad Djedid
//  Year      : 2025/2026
// =============================================================================

#include "QLearningMAC.h"

// ─────────────────────────────────────────────────────────────────────────────
//  Register module with OMNeT++
// ─────────────────────────────────────────────────────────────────────────────
Define_Module(QLearningMAC);


// =============================================================================
//  initialize()  —  called once at simulation start
// =============================================================================
void QLearningMAC::initialize(int stage)
{
    // ── Read parameters from omnetpp.ini / NED ────────────────────────────
    nodeId       = par("nodeId");
    txPowerDbm   = par("txPowerDbm").doubleValue();
    alpha        = par("alpha").doubleValue();
    gamma_       = par("gamma").doubleValue();
    epsilon      = par("epsilon0").doubleValue();
    epsilonMin   = par("epsilonMin").doubleValue();
    lambdaDecay  = par("lambdaDecay").doubleValue();
    cwMin        = par("cwMin");
    cwMax        = par("cwMax");
    macMaxRetries = par("macMaxRetries");

    // ── Initialize Q-table to zero ────────────────────────────────────────
    for (int s = 0; s < N_STATES; ++s)
        for (int a = 0; a < N_ACTIONS; ++a)
            qTable[s][a] = 0.0f;

    // ── Initialize state and action ───────────────────────────────────────
    curCW         = cwMin;
    curActionIdx  = 0;
    episode       = 0;
    retryCount    = 0;
    rng.seed(nodeId + 42);

    // ── Initialize statistics ─────────────────────────────────────────────
    stats.total_tx   = 0;
    stats.total_coll = 0;
    stats.busy_count = 0;
    stats.queue_len  = 0;
    stats.queue_max  = par("macBufferSize");

    curState = buildState(stats);

    // ── Output vectors for result collection ──────────────────────────────
    vecReward.setName("reward");
    vecCW.setName("selected_CW");
    vecCollRate.setName("collision_rate");
    vecEpsilon.setName("epsilon");

    // ── Schedule first backoff ────────────────────────────────────────────
    backoffTimer = nullptr;
    txTimer      = nullptr;
    ackTimer     = nullptr;

    scheduleBackoff(cwMin);

    EV_INFO << "[QLearningMAC] Node " << nodeId
            << " initialized. CW_min=" << cwMin
            << " CW_max=" << cwMax
            << " alpha=" << alpha
            << " gamma=" << gamma_
            << " epsilon=" << epsilon << "\n";
}


// =============================================================================
//  handleMessage()  —  main event handler
// =============================================================================
void QLearningMAC::handleMessage(cMessage *msg)
{
    // ── Backoff timer expired: attempt transmission ────────────────────────
    if (msg == backoffTimer) {
        delete backoffTimer;
        backoffTimer = nullptr;
        attemptTransmission();
    }

    // ── TX timer expired: observe outcome ─────────────────────────────────
    else if (msg == txTimer) {
        delete txTimer;
        txTimer = nullptr;
        // Wait for ACK within ackTimeout
        ackTimer = new cMessage("ackTimeout");
        scheduleAt(simTime() + ACK_TIMEOUT, ackTimer);
    }

    // ── ACK timeout: collision detected ───────────────────────────────────
    else if (msg == ackTimer) {
        delete ackTimer;
        ackTimer = nullptr;
        handleCollision();
    }

    // ── Packet from network layer ─────────────────────────────────────────
    else if (msg->arrivedOn("fromNetworkModule")) {
        handleUpperPacket(msg);
    }

    // ── Frame from radio (ACK or data) ────────────────────────────────────
    else if (msg->arrivedOn("fromRadioModule")) {
        handleRadioFrame(msg);
    }

    // ── Resource manager message ──────────────────────────────────────────
    else if (msg->arrivedOn("fromCommModuleResourceMgr")) {
        delete msg;
    }

    else {
        delete msg;
    }
}


// =============================================================================
//  attemptTransmission()  —  sense channel then transmit
// =============================================================================
void QLearningMAC::attemptTransmission()
{
    // Sense channel: if busy, freeze and retry
    bool channelBusy = senseChannel();
    if (channelBusy) {
        stats.recordTx(false, true);
        scheduleBackoff(curCW);   // restart backoff
        return;
    }

    // Channel idle: transmit
    EV_INFO << "[Node " << nodeId << "] Transmitting with CW=" << curCW << "\n";
    txTimer = new cMessage("txComplete");
    scheduleAt(simTime() + TX_DURATION, txTimer);
}


// =============================================================================
//  handleCollision()  —  no ACK received → collision
// =============================================================================
void QLearningMAC::handleCollision()
{
    retryCount++;
    bool channelBusy = (std::uniform_real_distribution<>(0,1)(rng) < 0.6);
    double energy    = E_TX * 1.5;   // extra energy from failed TX

    stats.recordTx(true, channelBusy);
    updateQueueLen(false);

    // Compute reward and learn
    double reward    = computeReward(false, curCW, energy);
    MACState nstate  = buildState(stats);
    updateQTable(curState, curActionIdx, reward, nstate);
    decayEpsilon();
    logMetrics(reward);

    // Select new CW and reschedule
    curState     = nstate;
    curActionIdx = selectAction(curState);
    curCW        = ACTIONS[curActionIdx];

    if (retryCount >= macMaxRetries) {
        // Drop packet after max retries
        EV_WARN << "[Node " << nodeId << "] Packet dropped after "
                << macMaxRetries << " retries.\n";
        retryCount = 0;
        stats.queue_len = std::max(0, stats.queue_len - 1);
        curCW = cwMin;
        curActionIdx = 0;
    }

    scheduleBackoff(curCW);
}


// =============================================================================
//  handleSuccess()  —  ACK received → successful TX
// =============================================================================
void QLearningMAC::handleSuccess()
{
    retryCount = 0;
    bool channelBusy = (std::uniform_real_distribution<>(0,1)(rng) < 0.2);
    double energy    = E_TX;

    stats.recordTx(false, channelBusy);
    updateQueueLen(true);

    // Compute reward and learn
    double reward    = computeReward(true, curCW, energy);
    MACState nstate  = buildState(stats);
    updateQTable(curState, curActionIdx, reward, nstate);
    decayEpsilon();
    logMetrics(reward);

    EV_INFO << "[Node " << nodeId << "] TX success. reward=" << reward
            << " epsilon=" << epsilon << "\n";

    // Select new CW for next packet
    curState     = nstate;
    curActionIdx = selectAction(curState);
    curCW        = ACTIONS[curActionIdx];

    scheduleBackoff(curCW);
}


// =============================================================================
//  handleUpperPacket()  —  packet arriving from network layer
// =============================================================================
void QLearningMAC::handleUpperPacket(cMessage *msg)
{
    // Push to queue if space available
    if (stats.queue_len < stats.queue_max) {
        stats.queue_len++;
        EV_INFO << "[Node " << nodeId << "] Packet enqueued. "
                << "Queue=" << stats.queue_len << "\n";
    } else {
        EV_WARN << "[Node " << nodeId << "] Queue full — packet dropped.\n";
    }
    delete msg;
}


// =============================================================================
//  handleRadioFrame()  —  frame arriving from radio module
// =============================================================================
void QLearningMAC::handleRadioFrame(cMessage *msg)
{
    std::string name = msg->getName();

    if (name == "ACK") {
        // Cancel ACK timeout and handle success
        if (ackTimer && ackTimer->isScheduled()) {
            cancelAndDelete(ackTimer);
            ackTimer = nullptr;
        }
        handleSuccess();
    } else {
        // Data frame for upper layers — forward
        send(msg, "toNetworkModule");
        return;
    }
    delete msg;
}


// =============================================================================
//  Q-Learning core methods
// =============================================================================

// ── selectAction() : ε-greedy policy ─────────────────────────────────────────
int QLearningMAC::selectAction(const MACState &s)
{
    int sIdx = s.toIndex();
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    if (dist(rng) < epsilon) {
        // Exploration: random action
        std::uniform_int_distribution<int> aDist(0, N_ACTIONS - 1);
        return aDist(rng);
    } else {
        // Exploitation: greedy action
        int best = 0;
        for (int a = 1; a < N_ACTIONS; ++a)
            if (qTable[sIdx][a] > qTable[sIdx][best])
                best = a;
        return best;
    }
}


// ── updateQTable() : Bellman update ──────────────────────────────────────────
void QLearningMAC::updateQTable(const MACState &s,
                                 int actionIdx,
                                 double reward,
                                 const MACState &ns)
{
    int sIdx  = s.toIndex();
    int nsIdx = ns.toIndex();

    // Max Q-value in next state
    float maxQ = qTable[nsIdx][0];
    for (int a = 1; a < N_ACTIONS; ++a)
        if (qTable[nsIdx][a] > maxQ)
            maxQ = qTable[nsIdx][a];

    // Bellman equation
    float tdTarget = (float)reward + (float)gamma_ * maxQ;
    float tdError  = tdTarget - qTable[sIdx][actionIdx];
    qTable[sIdx][actionIdx] += (float)alpha * tdError;
}


// ── computeReward() : multi-objective reward ─────────────────────────────────
double QLearningMAC::computeReward(bool success, int cw, double energy)
{
    double rSuccess = success ? R_SUCCESS : R_COLLISION;
    double rEnergy  = -BETA * energy;
    double rDelay   = -ALPHA_D * ((double)cw / cwMax);
    return W1 * rSuccess + W2 * rEnergy + W3 * rDelay;
}


// ── decayEpsilon() : exponential decay ───────────────────────────────────────
void QLearningMAC::decayEpsilon()
{
    episode++;
    epsilon = std::max(epsilonMin,
                       1.0 * std::exp(-lambdaDecay * episode));
}


// =============================================================================
//  Channel & backoff helpers
// =============================================================================

// ── scheduleBackoff() ─────────────────────────────────────────────────────────
void QLearningMAC::scheduleBackoff(int cw)
{
    if (backoffTimer && backoffTimer->isScheduled())
        cancelAndDelete(backoffTimer);

    std::uniform_int_distribution<int> dist(0, cw - 1);
    int slots = dist(rng);
    double delay = slots * SLOT_DURATION;   // 320 µs per IEEE 802.15.4 slot

    backoffTimer = new cMessage("backoff");
    scheduleAt(simTime() + delay, backoffTimer);

    EV_DETAIL << "[Node " << nodeId << "] Backoff: " << slots
              << " slots (" << delay * 1e3 << " ms), CW=" << cw << "\n";
}


// ── senseChannel() : CCA simulation ──────────────────────────────────────────
bool QLearningMAC::senseChannel()
{
    // Channel busy probability based on collision rate and queue
    double p_busy = stats.collRate() * 0.5 + stats.queueOcc() * 0.2;
    p_busy = std::min(p_busy, 0.90);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng) < p_busy;
}


// ── updateQueueLen() ──────────────────────────────────────────────────────────
void QLearningMAC::updateQueueLen(bool success)
{
    if (success && stats.queue_len > 0)
        stats.queue_len--;
    // Poisson arrivals: mean 0.15 packets per step
    std::poisson_distribution<int> poisson(0.15);
    int arrivals = poisson(rng);
    stats.queue_len = std::min(stats.queue_len + arrivals, stats.queue_max);
}


// =============================================================================
//  State helpers
// =============================================================================
int QLearningMAC::disc(double v, double lo, double hi)
{
    return (v < lo) ? 0 : (v < hi) ? 1 : 2;
}

MACState QLearningMAC::buildState(const MACStats &s)
{
    MACState st;
    st.delta = disc(s.collRate(),  0.20, 0.50);
    st.rho   = disc(s.busyRatio(), 0.30, 0.70);
    st.queue = disc(s.queueOcc(),  0.33, 0.66);
    return st;
}


// =============================================================================
//  Logging
// =============================================================================
void QLearningMAC::logMetrics(double reward)
{
    vecReward.record(reward);
    vecCW.record(curCW);
    vecCollRate.record(stats.collRate());
    vecEpsilon.record(epsilon);
}


// =============================================================================
//  finish()  —  called at end of simulation
// =============================================================================
void QLearningMAC::finish()
{
    EV_INFO << "\n[QLearningMAC] Node " << nodeId << " — Final Statistics:\n"
            << "  Total TX      : " << stats.total_tx    << "\n"
            << "  Collisions    : " << stats.total_coll  << "\n"
            << "  Collision Rate: " << stats.collRate()  << "\n"
            << "  Busy Count    : " << stats.busy_count  << "\n"
            << "  Final epsilon : " << epsilon           << "\n"
            << "  Episodes      : " << episode           << "\n"
            << "  Final CW      : " << curCW             << "\n";

    // Record final scalars
    recordScalar("final_collision_rate", stats.collRate());
    recordScalar("final_epsilon",        epsilon);
    recordScalar("total_episodes",       episode);
    recordScalar("final_CW",            curCW);

    // Cancel pending timers
    cancelAndDelete(backoffTimer);
    cancelAndDelete(txTimer);
    cancelAndDelete(ackTimer);
    backoffTimer = nullptr;
    txTimer      = nullptr;
    ackTimer     = nullptr;
}
