package AutomaticBrakingSystem {
    private import ScalarValues::*;

    enum def BrakeStatus {
        enum Released;
        enum Pressed;
        enum Modulating;
    }

    enum def RoadCondition {
        enum Dry;
        enum Wet;
        enum Icy;
        enum Snowy;
    }

    part def BrakePedal {
        attribute status : BrakeStatus;
        attribute pressure : Real;
    }

    part def TractionMonitor {
        attribute roadCondition : RoadCondition;
        attribute adhesion : Real;
    }

    part def BrakeController {
        attribute modulationFrequency : Real;
        attribute targetPressure : Real;
    }

    part def Vehicle {
        part brakePedal : BrakePedal;
        part tractionMonitor : TractionMonitor;
        part brakeController : BrakeController;
    }

    port def BrakeSignalPort {
        attribute pressure : Real;
    }

    port def TractionDataPort {
        attribute roadCondition : RoadCondition;
        attribute adhesion : Real;
    }

    part system : Vehicle {
        port brakeInput : BrakeSignalPort;
        port tractionInput : TractionDataPort;
        port brakeOutput : BrakeSignalPort;
    }

    connect system.brakePedal to system.brakeInput;
    connect system.tractionMonitor to system.tractionInput;
    connect system.brakeController to system.brakeOutput;
}