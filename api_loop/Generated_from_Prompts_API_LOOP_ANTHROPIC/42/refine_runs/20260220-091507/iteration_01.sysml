package VehicleStartStop {
    private import ScalarValues::*;

    enum def VehicleState {
        enum OFF;
        enum STARTING;
        enum OPERATING;
        enum STOPPING;
    }

    enum def BrakeState {
        enum RELEASED;
        enum DEPRESSED;
    }

    enum def ParkingBrakeState {
        enum APPLIED;
        enum RELEASED;
    }

    enum def SignalState {
        enum INACTIVE;
        enum ACTIVE;
    }

    item def StartSignal {
        attribute active : Boolean;
    }

    item def StopSignal {
        attribute active : Boolean;
    }

    item def PowerOnSignal {
        attribute active : Boolean;
    }

    item def ControllerStartSignal {
        attribute active : Boolean;
    }

    item def SelfCheckResult {
        attribute passed : Boolean;
    }

    port def VehicleInputPort {
        attribute startSignalReceived : Boolean;
        attribute stopSignalReceived : Boolean;
        attribute powerOnSignalReceived : Boolean;
        attribute brakePedalState : BrakeState;
    }

    port def VehicleOutputPort {
        attribute currentState : VehicleState;
        attribute parkingBrakeState : ParkingBrakeState;
        attribute controllerStartSignalSent : Boolean;
        attribute powerSupplyActive : Boolean;
        attribute selfCheckPassed : Boolean;
    }

    port def ControllerPort {
        attribute startCommandReceived : Boolean;
        attribute vehicleState : VehicleState;
    }

    part def BrakePedalSensor {
        attribute pedalDepressed : Boolean;
        attribute brakeState : BrakeState;
        port sensorOutput : VehicleInputPort;
    }

    part def ParkingBrakeActuator {
        attribute brakeApplied : Boolean;
        attribute parkingBrakeState : ParkingBrakeState;
        port actuatorInput : VehicleInputPort;
    }

    part def PowerSupply {
        attribute supplyActive : Boolean;
        attribute supplyVoltage : Real;
        port powerOutput : VehicleOutputPort;
    }

    part def SelfCheckModule {
        attribute checkComplete : Boolean;
        attribute checkPassed : Boolean;
        port checkOutput : VehicleOutputPort;
    }

    part def Controller {
        attribute startCommandReceived : Boolean;
        attribute vehicleOperating : Boolean;
        port controllerPort : ControllerPort;
    }

    part def StartStopControlUnit {
        attribute currentState : VehicleState = VehicleState::OFF;
        attribute brakePedalDepressed : Boolean = false;
        attribute startSignalReceived : Boolean = false;
        attribute stopSignalReceived : Boolean = false;
        attribute powerOnSignalReceived : Boolean = false;
        attribute controllerStartSignalSent : Boolean = false;
        attribute powerSupplyActive : Boolean = false;
        attribute selfCheckPassed : Boolean = false;
        attribute parkingBrakeApplied : Boolean = false;

        port inputPort : VehicleInputPort;
        port outputPort : VehicleOutputPort;
        port toController : ControllerPort;

        part brakeSensor : BrakePedalSensor;
        part parkingBrake : ParkingBrakeActuator;
        part powerSupply : PowerSupply;
        part selfCheck : SelfCheckModule;
        part controller : Controller;

        connect inputPort to brakeSensor.sensorOutput;
        connect inputPort to parkingBrake.actuatorInput;
        connect powerSupply.powerOutput to outputPort;
        connect selfCheck.checkOutput to outputPort;
        connect toController to controller.controllerPort;
    }

    part def VehicleStartStopSystem {
        attribute systemName : String = "VehicleStartStopSystem";
        attribute safetyEnabled : Boolean = true;
        attribute reliabilityEnabled : Boolean = true;

        part controlUnit : StartStopControlUnit;

        port systemInput : VehicleInputPort;
        port systemOutput : VehicleOutputPort;
        port controllerInterface : ControllerPort;

        connect systemInput to controlUnit.inputPort;
        connect controlUnit.outputPort to systemOutput;
        connect controlUnit.toController to controllerInterface;
    }
}