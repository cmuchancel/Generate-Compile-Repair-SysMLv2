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
        attribute engineState : EngineState = EngineState::OFF;
        attribute currentTorque : Real = 0.0;
        port fuelInput : FuelCommandPort;
        port torqueOutput : TorquePort;
        port stallSignal : StallSignalPort;
    }

    part def TorqueAmplifier {
        attribute amplificationFactor : Real = 2.0;
        port inputTorque : TorquePort;
        port outputTorque : TorquePort;
    }

    part def Transmission {
        port inputTorque : TorquePort;
        port outputTorque : TorquePort;
    }

    part def Differential {
        port inputTorque : TorquePort;
        port leftWheelTorque : TorquePort;
        port rightWheelTorque : TorquePort;
    }

    part def PowertrainController {
        attribute transmissionState : TorqueTransmissionState = TorqueTransmissionState::IDLE;
        port fuelCommand : FuelCommandPort;
        port stallDetection : StallSignalPort;
        port engineControl : EngineControlPort;
    }

    item def FuelCommandItem;
    item def TorqueItem;
    item def StallSignalItem;
    item def EngineControlItem;

    port def FuelCommandPort;
    port def TorquePort;
    port def StallSignalPort;
    port def EngineControlPort;

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