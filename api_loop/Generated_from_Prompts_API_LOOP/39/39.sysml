package PowerManagement {
    private import ScalarValues::*;

    enum def OperatingState {
        enum Off;
        enum Start;
        enum On;
    }

    enum def HealthState {
        enum Unknown;
        enum Healthy;
        enum Degraded;
        enum Fault;
    }

    port def ControlSignals {
        attribute startSignal: Boolean;
        attribute onSignal: Boolean;
        attribute offSignal: Boolean;
    }

    part def VehiclePowerManager {
        attribute operatingState: OperatingState = OperatingState::Off;
        attribute healthState: HealthState = HealthState::Unknown;
        port control: ControlSignals;
    }

    part def Vehicle {
        part powerManager: VehiclePowerManager;
    }
}