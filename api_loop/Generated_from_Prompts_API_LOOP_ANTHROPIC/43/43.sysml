package VehicleStartStopControl {
    private import ScalarValues::*;

    enum def VehicleState {
        enum OFF;
        enum STARTING;
        enum STARTED;
        enum SHUTDOWN;
    }

    enum def BrakeState {
        enum RELEASED;
        enum DEPRESSED;
    }

    enum def ParkingBrakeState {
        enum DISENGAGED;
        enum ENGAGED;
    }

    enum def SignalState {
        enum INACTIVE;
        enum ACTIVE;
    }

    item def VehicleStartSignal {
        attribute signalState : SignalState;
    }

    item def VehicleOnSignal {
        attribute signalState : SignalState;
    }

    item def ShutdownSignal {
        attribute signalState : SignalState;
    }

    item def StartControlSignal {
        attribute signalState : SignalState;
    }

    item def PowerSupplySignal {
        attribute signalState : SignalState;
    }

    item def SelfCheckSignal {
        attribute signalState : SignalState;
    }

    port def DriverInputPort {
        attribute brakeState : BrakeState;
        attribute vehicleStartSignal : SignalState;
        attribute vehicleOnSignal : SignalState;
        attribute shutdownSignal : SignalState;
    }

    port def ControllerOutputPort {
        attribute startControlSignal : SignalState;
        attribute powerSupplySignal : SignalState;
        attribute selfCheckSignal : SignalState;
        attribute parkingBrakeCommand : ParkingBrakeState;
    }

    port def VehicleControllerPort {
        attribute receivedStartControl : SignalState;
        attribute vehicleState : VehicleState;
    }

    part def DriverInputUnit {
        attribute currentBrakeState : BrakeState = BrakeState::RELEASED;
        attribute currentVehicleStartSignal : SignalState = SignalState::INACTIVE;
        attribute currentVehicleOnSignal : SignalState = SignalState::INACTIVE;
        attribute currentShutdownSignal : SignalState = SignalState::INACTIVE;

        port driverOut : DriverInputPort;
    }

    part def StartStopController {
        attribute currentVehicleState : VehicleState = VehicleState::OFF;
        attribute currentParkingBrake : ParkingBrakeState = ParkingBrakeState::DISENGAGED;
        attribute startControlOutput : SignalState = SignalState::INACTIVE;
        attribute powerSupplyOutput : SignalState = SignalState::INACTIVE;
        attribute selfCheckOutput : SignalState = SignalState::INACTIVE;

        port controllerIn : DriverInputPort;
        port controllerOut : ControllerOutputPort;
        port vehicleControllerLink : VehicleControllerPort;
    }

    part def VehicleController {
        attribute vehicleState : VehicleState = VehicleState::OFF;
        attribute receivedStartControl : SignalState = SignalState::INACTIVE;
        attribute powerActive : SignalState = SignalState::INACTIVE;
        attribute selfCheckPassed : SignalState = SignalState::INACTIVE;
        attribute parkingBrakeEngaged : ParkingBrakeState = ParkingBrakeState::DISENGAGED;

        port vehicleIn : VehicleControllerPort;
        port vehicleControlOut : ControllerOutputPort;
    }

    part def VehicleStartStopSystem {
        part driverUnit : DriverInputUnit;
        part controller : StartStopController;
        part vehicleCtrl : VehicleController;

        connect driverUnit.driverOut to controller.controllerIn;
        connect controller.vehicleControllerLink to vehicleCtrl.vehicleIn;
        connect vehicleCtrl.vehicleControlOut to controller.controllerOut;
    }
}