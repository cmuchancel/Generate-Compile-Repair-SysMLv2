package AutomotivePowertrainControl {
    private import ScalarValues::*;

    enum def EngineState {
        enum OFF;
        enum STARTING;
        enum RUNNING;
        enum STALLING;
    }

    enum def TorqueTransmissionState {
        enum IDLE;
        enum AMPLIFYING;
        enum TRANSMITTING;
        enum DISTRIBUTING;
    }

    part def Engine {
        attribute state : EngineState = EngineState::OFF;
        attribute currentTorque : Real = 0.0;
        port fuelInput : FuelCommand;
        port torqueOutput : Torque;
        port stallSignal : StallSignal;
    }

    part def TorqueAmplifier {
        attribute amplificationFactor : Real = 2.0;
        port inputTorque : Torque;
        port outputTorque : Torque;
    }

    part def Transmission {
        port inputTorque : Torque;
        port outputTorque : Torque;
    }

    part def Differential {
        port inputTorque : Torque;
        port leftWheelTorque : Torque;
        port rightWheelTorque : Torque;
    }

    part def PowertrainController {
        attribute transmissionState : TorqueTransmissionState = TorqueTransmissionState::IDLE;
        port fuelCommand : FuelCommand;
        port stallDetection : StallSignal;
        port engineControl : EngineControl;
    }

    item def FuelCommand;
    item def Torque;
    item def StallSignal;
    item def EngineControl;

    port def FuelCommand;
    port def Torque;
    port def StallSignal;
    port def EngineControl;

    part powertrainSystem {
        part engine : Engine;
        part amplifier : TorqueAmplifier;
        part transmission : Transmission;
        part differential : Differential;
        part controller : PowertrainController;

        connect controller.fuelCommand to engine.fuelInput;
        connect controller.stallDetection to engine.stallSignal;
        connect controller.engineControl to engine;
        connect engine.torqueOutput to amplifier.inputTorque;
        connect amplifier.outputTorque to transmission.inputTorque;
        connect transmission.outputTorque to differential.inputTorque;
    }
}