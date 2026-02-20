package PowertrainControlLogic {
    private import ScalarValues::*;

    item def FuelCommand;
    item def Torque;
    item def StallSignal;
    item def StartCommand;
    item def StopCommand;

    port def FuelCommandPort { item FuelCommand; }
    port def TorquePort { item Torque; }
    port def StallSignalPort { item StallSignal; }
    port def StartCommandPort { item StartCommand; }
    port def StopCommandPort { item StopCommand; }

    part def Engine {
        port fuelIn : FuelCommandPort;
        port startIn : StartCommandPort;
        port stopIn : StopCommandPort;
        port stallOut : StallSignalPort;
        port torqueOut : TorquePort;
        attribute started : Boolean;
    }

    part def TorqueAmplifier {
        port torqueIn : TorquePort;
        port torqueOut : TorquePort;
    }

    part def Transmission {
        port torqueIn : TorquePort;
        port torqueOut : TorquePort;
    }

    part def Distribution {
        port torqueIn : TorquePort;
        port torqueOut : TorquePort;
    }

    part def DriveWheels {
        port torqueIn : TorquePort;
    }

    part def StallMonitor {
        port stallIn : StallSignalPort;
        port stopOut : StopCommandPort;
    }

    part def PowertrainControlSystem {
        port fuelCmdIn : FuelCommandPort;
        port startCmdIn : StartCommandPort;
        port stopCmdIn : StopCommandPort;
        port wheelTorqueOut : TorquePort;

        part engine : Engine;
        part amplifier : TorqueAmplifier;
        part transmission : Transmission;
        part distribution : Distribution;
        part wheels : DriveWheels;
        part stallMonitor : StallMonitor;

        connect fuelCmdIn to engine.fuelIn;
        connect startCmdIn to engine.startIn;
        connect stopCmdIn to engine.stopIn;
        connect engine.torqueOut to amplifier.torqueIn;
        connect amplifier.torqueOut to transmission.torqueIn;
        connect transmission.torqueOut to distribution.torqueIn;
        connect distribution.torqueOut to wheels.torqueIn;
        connect distribution.torqueOut to wheelTorqueOut;
        connect engine.stallOut to stallMonitor.stallIn;
        connect stallMonitor.stopOut to engine.stopIn;
    }
}