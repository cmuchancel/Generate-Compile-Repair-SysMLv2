package VehicleBehaviorSystem {
    private import ScalarValues::*;

    enum def VehicleState {
        enum SHUTDOWN;
        enum STARTUP;
        enum NORMAL;
        enum DEGRADED;
        enum MAINTENANCE;
    }

    enum def SignalType {
        enum START_SIGNAL;
        enum WARNING_SIGNAL;
        enum SHUTDOWN_SIGNAL;
    }

    part def Vehicle {
        attribute state : VehicleState = VehicleState::SHUTDOWN;
        attribute brakePressed : Boolean = false;
        attribute startSignalReceived : Boolean = false;
        attribute temperature : Real;
        attribute fuelLevel : Real;
        attribute maintenanceInterval : Real;
        attribute temperatureThreshold : Real = 100.0;

        port controlPort : VehicleControlPort;
        port healthPort : HealthMonitoringPort;
    }

    part def VehicleController {
        port vehiclePort : VehicleControlPort;
        port warningPort : WarningPort;
    }

    port def VehicleControlPort {
        in startSignal : SignalType;
        out shutdownSignal : SignalType;
    }

    port def HealthMonitoringPort {
        out warningSignal : SignalType;
    }

    port def WarningPort {
        in warningSignal : SignalType;
    }

    item def ControlSignal {
        attribute signalType : SignalType;
    }

    part vehicle : Vehicle;
    part controller : VehicleController;

    connect vehicle.controlPort to controller.vehiclePort;
    connect vehicle.healthPort to controller.warningPort;
}