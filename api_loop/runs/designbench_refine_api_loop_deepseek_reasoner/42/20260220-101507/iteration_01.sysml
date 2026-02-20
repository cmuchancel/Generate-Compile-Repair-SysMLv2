package VehicleStartStopControl {
    private import ScalarValues::*;

    enum def VehicleState {
        enum Off;
        enum Starting;
        enum Operating;
    }

    port def ControlSignalPort {
        attribute signalReceived : Boolean = false;
    }

    port def PowerPort {
        attribute powerOn : Boolean = false;
    }

    port def BrakeSignalPort {
        attribute brakeDepressed : Boolean = false;
    }

    port def ParkingBrakePort {
        attribute parkingBrakeApplied : Boolean = false;
    }

    part def Controller {
        port controlInput : ControlSignalPort;
    }

    part def VehicleSystem {
        port startSignalIn : ControlSignalPort;
        port stopSignalIn : ControlSignalPort;
        port powerOnSignalIn : ControlSignalPort;
        port brakeSignalIn : BrakeSignalPort;
        port controllerOut : ControlSignalPort;
        port powerOut : PowerPort;
        port parkingBrakeOut : ParkingBrakePort;

        attribute currentState : VehicleState = VehicleState::Off;
        attribute brakeDepressed : Boolean = false;
        attribute startSignalReceived : Boolean = false;
        attribute stopSignalReceived : Boolean = false;
        attribute powerOnSignalReceived : Boolean = false;
        attribute selfCheckCompleted : Boolean = false;
        attribute powerSupplied : Boolean = false;
        attribute parkingBrakeApplied : Boolean = false;
    }

    part vehicleSystem : VehicleSystem;
    part controller : Controller;

    connect vehicleSystem.startSignalIn to vehicleSystem.brakeSignalIn;
    connect vehicleSystem.controllerOut to controller.controlInput;
    connect vehicleSystem.powerOut to vehicleSystem.powerOut;
    connect vehicleSystem.parkingBrakeOut to vehicleSystem.parkingBrakeOut;
}