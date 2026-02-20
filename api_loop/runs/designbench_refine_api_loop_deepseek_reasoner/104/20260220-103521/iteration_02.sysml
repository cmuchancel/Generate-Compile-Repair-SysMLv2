package PowertrainSystem {
    private import ScalarValues::*;

    item def Torque {
        attribute value : Real;
    }

    enum def EngineState {
        enum STOPPED;
        enum STARTED;
    }

    port def TorquePort {
        attribute torque : Torque;
    }

    port def ControlPort {
        attribute command : Real;
    }

    port def SwitchPort {
        attribute engineState : EngineState;
    }

    part def Engine {
        port fuelIn : ControlPort;
        port switchIn : SwitchPort;
        port torqueOut : TorquePort;
        attribute currentState : EngineState = EngineState::STOPPED;
        attribute torque : Torque;
    }

    part def Transmission {
        port torqueIn : TorquePort;
        port torqueOut : TorquePort;
        attribute ratio : Real = 1.0;
        attribute torque : Torque;
    }

    part def DriveShaft {
        port torqueIn : TorquePort;
        port leftOut : TorquePort;
        port rightOut : TorquePort;
        attribute torque : Torque;
    }

    part def Wheel {
        port torqueIn : TorquePort;
        attribute torque : Torque;
    }

    part def Powertrain {
        part engine : Engine;
        part transmission : Transmission;
        part driveshaft : DriveShaft;
        part leftWheel : Wheel;
        part rightWheel : Wheel;

        connect engine.torqueOut to transmission.torqueIn;
        connect transmission.torqueOut to driveshaft.torqueIn;
        connect driveshaft.leftOut to leftWheel.torqueIn;
        connect driveshaft.rightOut to rightWheel.torqueIn;
    }
}