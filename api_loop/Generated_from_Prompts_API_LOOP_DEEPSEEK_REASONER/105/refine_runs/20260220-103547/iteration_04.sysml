package PowertrainControlSystem {
    private import ScalarValues::*;

    enum def EngineState {
        enum Off;
        enum Starting;
        enum Running;
        enum Stopping;
    }

    enum def TorqueState {
        enum Zero;
        enum Generated;
        enum Amplified;
        enum Transmitted;
        enum Distributed;
    }

    item def FuelControlCommand {
        attribute level : Real;
    }

    item def TorqueSignal {
        attribute value : Real;
    }

    item def StallSignal {
        attribute active : Boolean;
    }

    port def ControlPort {
        attribute command : FuelControlCommand;
        attribute stall : StallSignal;
    }

    port def TorquePort {
        attribute torque : TorqueSignal;
    }

    part def FuelController {
        port control : ControlPort;
        attribute currentCommand : FuelControlCommand;
    }

    part def Engine {
        port input : ControlPort;
        port output : TorquePort;
        attribute engineState : EngineState;
        attribute generatedTorque : Real;
    }

    part def TorqueAmplifier {
        port input : TorquePort;
        port output : TorquePort;
        attribute amplificationFactor : Real = 1.5;
    }

    part def Transmission {
        port input : TorquePort;
        port output : TorquePort;
        attribute gearRatio : Real = 4.2;
    }

    part def Differential {
        port input : TorquePort;
        port leftOutput : TorquePort;
        port rightOutput : TorquePort;
        attribute splitRatio : Real = 0.5;
    }

    part def Wheel {
        port input : TorquePort;
        attribute deliveredTorque : Real;
    }

    part def Powertrain {
        part fuelController : FuelController;
        part engine : Engine;
        part torqueAmplifier : TorqueAmplifier;
        part transmission : Transmission;
        part differential : Differential;
        part leftWheel : Wheel;
        part rightWheel : Wheel;

        port systemControl : ControlPort;
        port leftTorqueOutput : TorquePort;
        port rightTorqueOutput : TorquePort;

        connect systemControl to fuelController.control;
        connect fuelController.control to engine.input;
        connect engine.output to torqueAmplifier.input;
        connect torqueAmplifier.output to transmission.input;
        connect transmission.output to differential.input;
        connect differential.leftOutput to leftWheel.input;
        connect differential.rightOutput to rightWheel.input;
        connect leftWheel.input to leftTorqueOutput;
        connect rightWheel.input to rightTorqueOutput;
    }
}