package AutomotivePowertrain {
    private import ScalarValues::*;

    enum def EngineState {
        enum OFF;
        enum STARTING;
        enum RUNNING;
        enum STOPPING;
    }

    part def Engine {
        attribute currentState : EngineState = EngineState::OFF;
        attribute fuelCommand : Real;
        attribute torqueOutput : Real;
        port fuelInput : FuelCommandPort;
        port torqueOutputPort : TorquePort;
    }

    part def Transmission {
        attribute gearRatio : Real;
        port inputTorque : TorquePort;
        port outputTorque : TorquePort;
    }

    part def Driveshaft {
        port inputTorque : TorquePort;
        port leftOutput : TorquePort;
        port rightOutput : TorquePort;
    }

    part def Wheel {
        port torqueInput : TorquePort;
    }

    part def PowertrainController {
        port engineControl : EngineControlPort;
        port userInput : UserCommandPort;
    }

    port def FuelCommandPort;
    port def TorquePort;
    port def EngineControlPort;
    port def UserCommandPort;

    part powertrainSystem {
        part engine : Engine;
        part transmission : Transmission;
        part driveshaft : Driveshaft;
        part leftWheel : Wheel;
        part rightWheel : Wheel;
        part controller : PowertrainController;

        connect controller.engineControl to engine.fuelInput;
        connect engine.torqueOutputPort to transmission.inputTorque;
        connect transmission.outputTorque to driveshaft.inputTorque;
        connect driveshaft.leftOutput to leftWheel.torqueInput;
        connect driveshaft.rightOutput to rightWheel.torqueInput;
        connect controller.userInput to engine.fuelInput;
    }
}