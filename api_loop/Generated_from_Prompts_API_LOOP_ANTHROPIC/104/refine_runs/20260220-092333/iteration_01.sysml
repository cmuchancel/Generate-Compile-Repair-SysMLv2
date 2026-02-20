package AutomotivePowertrain {
    private import ScalarValues::*;

    enum def EngineState {
        enum OFF;
        enum RUNNING;
    }

    enum def TransmissionState {
        enum IDLE;
        enum ENGAGED;
    }

    item def FuelCommand {
        attribute level : Real;
    }

    item def ShutdownCommand {
        attribute requested : Boolean;
    }

    item def Torque {
        attribute value : Real;
    }

    item def Power {
        attribute value : Real;
    }

    port def FuelCommandPort {
        attribute level : Real;
    }

    port def ShutdownPort {
        attribute requested : Boolean;
    }

    port def TorquePort {
        attribute value : Real;
    }

    port def PowerPort {
        attribute value : Real;
    }

    part def UserInterface {
        port fuelOut : FuelCommandPort;
        port shutdownOut : ShutdownPort;
    }

    part def Engine {
        attribute state : EngineState;
        attribute engineTorque : Real;
        attribute fuelLevel : Real;

        port fuelIn : FuelCommandPort;
        port shutdownIn : ShutdownPort;
        port torqueOut : TorquePort;
    }

    part def Transmission {
        attribute state : TransmissionState;
        attribute amplifiedTorque : Real;
        attribute gearRatio : Real;

        port torqueIn : TorquePort;
        port torqueOut : TorquePort;
    }

    part def Driveshaft {
        attribute transmittedPower : Real;

        port torqueIn : TorquePort;
        port powerOut : PowerPort;
    }

    part def WheelLeft {
        attribute drivingForce : Real;

        port powerIn : PowerPort;
    }

    part def WheelRight {
        attribute drivingForce : Real;

        port powerIn : PowerPort;
    }

    part def Differential {
        attribute inputPower : Real;
        attribute leftPower : Real;
        attribute rightPower : Real;

        port powerIn : PowerPort;
        port leftOut : PowerPort;
        port rightOut : PowerPort;
    }

    part def PowertrainSystem {
        part ui : UserInterface;
        part engine : Engine;
        part transmission : Transmission;
        part driveshaft : Driveshaft;
        part differential : Differential;
        part wheelLeft : WheelLeft;
        part wheelRight : WheelRight;

        connect ui.fuelOut to engine.fuelIn;
        connect ui.shutdownOut to engine.shutdownIn;
        connect engine.torqueOut to transmission.torqueIn;
        connect transmission.torqueOut to driveshaft.torqueIn;
        connect driveshaft.powerOut to differential.powerIn;
        connect differential.leftOut to wheelLeft.powerIn;
        connect differential.rightOut to wheelRight.powerIn;
    }
}