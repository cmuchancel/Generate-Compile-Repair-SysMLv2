package VehiclePowerManagement {
    private import ScalarValues::*;

    enum def VehicleState {
        enum Off;
        enum Start;
        enum On;
    }

    enum def ControlSignal {
        enum StartSignal;
        enum OnSignal;
        enum OffSignal;
    }

    enum def HealthState {
        enum Normal;
        enum Warning;
        enum Critical;
    }

    item def PowerSignalItem {
        attribute signal : ControlSignal;
    }

    item def HealthDataItem {
        attribute health : HealthState;
    }

    port def PowerControlPort {
        attribute currentSignal : ControlSignal;
    }

    port def HealthMonitorPort {
        attribute healthStatus : HealthState;
    }

    part def PowerController {
        attribute currentState : VehicleState = VehicleState::Off;
        attribute lastSignal : ControlSignal;

        port controlInput : PowerControlPort;
        port healthOutput : HealthMonitorPort;
    }

    part def VehicleHealthMonitor {
        attribute healthState : HealthState = HealthState::Normal;

        port healthInput : HealthMonitorPort;
    }

    part def VehiclePowerSystem {
        attribute vehicleState : VehicleState = VehicleState::Off;
        attribute offToStartEnabled : Boolean = true;
        attribute startToOnEnabled : Boolean = true;
        attribute onToOffEnabled : Boolean = true;

        port systemControlPort : PowerControlPort;
        port systemHealthPort : HealthMonitorPort;

        part controller : PowerController;
        part healthMonitor : VehicleHealthMonitor;

        connect controller.healthOutput to healthMonitor.healthInput;
    }

    part vehiclePowerSystemInstance : VehiclePowerSystem;
}