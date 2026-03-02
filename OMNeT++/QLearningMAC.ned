// =============================================================================
//  QLearningMAC.ned
//  NED Module Description - Self-Adaptive CW for WSN Using Q-Learning
//  OMNeT++ 6.0 / Castalia 3.3
//
//  Thesis : Self-Adaptive Contention Window for WSN Using Q-Learning
//  University : Ammar Telidji University of Laghouat
//  Authors   : Makhloufi Mohammed Abderrahim - Makhloufi Khadidja
//  Supervisor : Dr. Lakhdar Oulad Djedid
// =============================================================================

package node.communication.mac.qlearningMAC;

simple QLearningMAC like node.communication.mac.iMac
{
    parameters:
        int    nodeId        = default(0);

        double alpha         = default(0.1);
        double gamma         = default(0.9);
        double epsilon0      = default(1.0);
        double epsilonMin    = default(0.01);
        double lambdaDecay   = default(0.005);

        double w1            = default(0.5);
        double w2            = default(0.3);
        double w3            = default(0.2);

        int    cwMin         = default(8);
        int    cwMax         = default(256);
        double txPowerDbm    = default(0.0);

        int    macBufferSize    = default(32);
        int    macFrameOverhead = default(5);
        int    macMaxRetries    = default(3);

        bool   collectTraceInfo = default(true);
        string applicationID    = default("qlCWApp");

        @display("i=block/mac");

        @signal[reward](type=double);
        @signal[selected_CW](type=int);
        @signal[collision_rate](type=double);
        @signal[epsilon](type=double);

        @statistic[reward](title="QL Reward";
            source=reward; record=vector,mean,max);
        @statistic[selected_CW](title="Selected CW";
            source=selected_CW; record=vector,mean);
        @statistic[collision_rate](title="Collision Rate";
            source=collision_rate; record=vector,mean);
        @statistic[epsilon](title="Epsilon";
            source=epsilon; record=vector,last);

    gates:
        output toNetworkModule;
        output toRadioModule;
        input  fromNetworkModule;
        input  fromRadioModule;
        input  fromCommModuleResourceMgr;
}


network QL_WSN_Network
{
    parameters:
        int    numNodes = default(20);
        double fieldX   = default(100);
        double fieldY   = default(100);

    submodules:
        node[numNodes] : SensorNode {
            @display("p=50,50");
        }

    connections allowunconnected:
}
