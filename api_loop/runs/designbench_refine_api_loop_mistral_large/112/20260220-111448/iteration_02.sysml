package VehicleControlSystem {
    private import ScalarValues::*;

    enum def VehicleState {
        enum Off;
        enum Starting;
        enum Running;
        enum Degraded;
        enum Maintenance;
    }

    enum def HealthStatus {
        enum Normal;
        enum OverTemperature;
    }

    part def VehicleA {
        attribute currentState : VehicleState = VehicleState::Off;
        attribute health : HealthStatus = HealthStatus::Normal;
        attribute temperature : Real;
        attribute safetyThreshold : Real = 100.0;
        attribute brakePressed : Boolean = false;
        attribute startSignal : Boolean = false;
        attribute vehicleOnSignal : Boolean = false;
        attribute restoreToNormalSignal : Boolean = false;
        attribute parkingBrakeApplied : Boolean = false;

        port startUpPort : StartUpPort;
        port healthMonitorPort : HealthMonitorPort;
        port controllerPort : ControllerPort;
    }

    part def VehicleController {
        attribute controllerState : VehicleState = VehicleState::Off;

        port controlPort : ControllerPort;
    }

    port def StartUpPort {
        attribute brakeInput : Boolean;
        attribute startInput : Boolean;
        attribute vehicleOnOutput : Boolean;
    }

    port def HealthMonitorPort {
        attribute temperatureInput : Real;
        attribute healthOutput : HealthStatus;
        attribute overTempOutput : Boolean;
    }

    port def ControllerPort {
        attribute stateInput : VehicleState;
        attribute stateOutput : VehicleState;
        attribute restoreSignal : Boolean;
    }

    part vehicle : VehicleA;
    part controller : VehicleController;

    connect vehicle.startUpPort to controller.controlPort;
    connect vehicle.healthMonitorPort to controller.controlPort;
    connect controller.controlPort to vehicle.controllerPort;
}