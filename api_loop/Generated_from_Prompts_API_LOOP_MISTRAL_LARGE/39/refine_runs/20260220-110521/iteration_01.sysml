package VehiclePowerManagement {
    private import ScalarValues::*;

    enum def OperatingState {
        enum Off;
        enum Start;
        enum On;
    }

    part def Vehicle {
        attribute currentState : OperatingState = OperatingState::Off;
        attribute healthState : String;
    }

    port def ControlSignal {
        attribute signal : String;
    }

    part def User {
        port startSignal : ControlSignal;
        port onSignal : ControlSignal;
        port offSignal : ControlSignal;
    }

    part def PowerManager {
        port startInput : ControlSignal;
        port onInput : ControlSignal;
        port offInput : ControlSignal;
        port stateOutput : ControlSignal;
    }

    item def StartSignal {
        attribute value : String = "start";
    }

    item def OnSignal {
        attribute value : String = "on";
    }

    item def OffSignal {
        attribute value : String = "off";
    }

    part vehicle : Vehicle;
    part user : User;
    part powerManager : PowerManager;

    connect user.startSignal to powerManager.startInput;
    connect user.onSignal to powerManager.onInput;
    connect user.offSignal to powerManager.offInput;
    connect powerManager.stateOutput to vehicle;
}