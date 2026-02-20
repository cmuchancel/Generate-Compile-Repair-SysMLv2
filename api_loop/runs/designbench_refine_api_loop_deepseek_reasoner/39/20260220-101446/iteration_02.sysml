package PowerManagement {
    private import ScalarValues::*;

    enum def OperatingState {
        enum Off;
        enum Start;
        enum On;
    }

    port def SignalPort;
    port def HealthPort;

    part def Vehicle {
        attribute currentState : OperatingState = OperatingState::Off;
        port startSignal : SignalPort;
        port onSignal : SignalPort;
        port offSignal : SignalPort;
        port healthMonitor : HealthPort;
    }

    part def User {
        port startSignal : SignalPort;
        port onSignal : SignalPort;
        port offSignal : SignalPort;
    }

    part def HealthManager {
        port healthMonitor : HealthPort;
    }

    part vehicle : Vehicle;
    part user : User;
    part healthManager : HealthManager;

    connect user.startSignal to vehicle.startSignal;
    connect user.onSignal to vehicle.onSignal;
    connect user.offSignal to vehicle.offSignal;
    connect vehicle.healthMonitor to healthManager.healthMonitor;
}