package VehicleSystem {
    private import ScalarValues::*;

    enum def VehicleStatus {
        enum Shutdown;
        enum Starting;
        enum Running;
        enum DegradedMode;
    }

    item def StartSignal;
    item def WarningSignal;

    port def ControlPort;
    port def StatusPort;

    part def Vehicle {
        port control : ControlPort;
        port status : StatusPort;
        attribute temperature : Real;
        attribute fuel : Real = 100.0;
        attribute maintenanceInterval : Integer = 90;
        attribute overheatThreshold : Real = 120.0;
        attribute vehicleStatus : VehicleStatus = VehicleStatus::Shutdown;
        attribute brakePedalPressed : Boolean = false;
        attribute startSignalReceived : Boolean = false;
    }

    part def VehicleController {
        port control : ControlPort;
        port status : StatusPort;
    }

    part vehicle : Vehicle;
    part controller : VehicleController;

    connect vehicle.control to controller.control;
    connect vehicle.status to controller.status;
}