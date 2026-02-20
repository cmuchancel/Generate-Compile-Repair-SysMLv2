package AutomotivePowertrain {
    private import ScalarValues::*;

    enum def EngineState {
        enum STOPPED;
        enum STARTING;
        enum RUNNING;
        enum STALLED;
    }

    enum def TransmissionState {
        enum IDLE;
        enum TRANSMITTING;
        enum HALTED;
    }

    enum def PowerFlowState {
        enum INACTIVE;
        enum ACTIVE;
        enum DISTRIBUTED;
    }

    item def FuelControlCommand {
        attribute commandId : Integer;
        attribute fuelLevel : Real;
    }

    item def StallSignal {
        attribute signalId : Integer;
        attribute detected : Boolean;
    }

    item def TorqueSignal {
        attribute torqueValue : Real;
    }

    item def PowerSignal {
        attribute powerValue : Real;
    }

    port def FuelCommandPort {
        attribute commandReceived : Boolean;
    }

    port def StallSignalPort {
        attribute stallDetected : Boolean;
    }

    port def TorqueOutputPort {
        attribute torque : Real;
    }

    port def PowerOutputPort {
        attribute power : Real;
    }

    port def WheelOutputPort {
        attribute wheelTorque : Real;
    }

    part def EngineController {
        attribute engineState : EngineState;
        attribute currentTorque : Real;
        attribute engineRunning : Boolean;

        port fuelIn : FuelCommandPort;
        port stallIn : StallSignalPort;
        port torqueOut : TorqueOutputPort;
    }

    part def TransmissionSystem {
        attribute transmissionState : TransmissionState;
        attribute amplificationFactor : Real;
        attribute amplifiedTorque : Real;

        port torqueIn : TorqueOutputPort;
        port powerOut : PowerOutputPort;
    }

    part def DriveShaft {
        attribute powerFlowState : PowerFlowState;
        attribute transmittedPower : Real;

        port powerIn : PowerOutputPort;
        port torqueTransmitted : TorqueOutputPort;
    }

    part def DifferentialUnit {
        attribute distributionRatio : Real;
        attribute leftWheelTorque : Real;
        attribute rightWheelTorque : Real;

        port torqueIn : TorqueOutputPort;
        port leftWheelOut : WheelOutputPort;
        port rightWheelOut : WheelOutputPort;
    }

    part def DriveWheel {
        attribute wheelId : Integer;
        attribute outputTorque : Real;
        attribute wheelActive : Boolean;

        port wheelIn : WheelOutputPort;
    }

    part def StallMonitor {
        attribute monitoring : Boolean;
        attribute stallEventCount : Integer;

        port stallOut : StallSignalPort;
    }

    part def PowertrainControlUnit {
        attribute systemActive : Boolean;
        attribute fuelCommandReceived : Boolean;
        attribute powerDeliveryEfficiency : Real;

        port fuelCommandIn : FuelCommandPort;
        port stallSignalIn : StallSignalPort;
        port mainTorqueOut : TorqueOutputPort;
        port mainPowerOut : PowerOutputPort;
        port finalWheelOut : WheelOutputPort;
    }

    part def AutomotivePowertrainSystem {
        attribute systemId : Integer;
        attribute overallState : EngineState;
        attribute totalOutputTorque : Real;
        attribute systemEnabled : Boolean;

        part controlUnit : PowertrainControlUnit;
        part engine : EngineController;
        part transmission : TransmissionSystem;
        part driveShaft : DriveShaft;
        part differential : DifferentialUnit;
        part leftWheel : DriveWheel;
        part rightWheel : DriveWheel;
        part stallMonitor : StallMonitor;

        connect controlUnit.fuelCommandIn to engine.fuelIn;
        connect stallMonitor.stallOut to engine.stallIn;
        connect stallMonitor.stallOut to controlUnit.stallSignalIn;
        connect engine.torqueOut to transmission.torqueIn;
        connect transmission.powerOut to driveShaft.powerIn;
        connect driveShaft.torqueTransmitted to differential.torqueIn;
        connect differential.leftWheelOut to leftWheel.wheelIn;
        connect differential.rightWheelOut to rightWheel.wheelIn;
    }
}