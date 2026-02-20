package PowerManagement {
    private import ScalarValues::*;
    
    enum def OperatingState {
        Off,
        Start,
        On
    }

    part def Vehicle {
        attribute currentState : OperatingState = OperatingState::Off;
        port in startSignal;
        port in onSignal;
        port in offSignal;
        port out healthMonitor;
    }

    part def User {
        port out startSignal;
        port out onSignal;
        port out offSignal;
    }

    part def HealthManager {
        port in healthMonitor;
    }

    part vehicle : Vehicle;
    part user : User;
    part healthManager : HealthManager;

    connect user.startSignal to vehicle.startSignal;
    connect user.onSignal to vehicle.onSignal;
    connect user.offSignal to vehicle.offSignal;
    connect vehicle.healthMonitor to healthManager.healthMonitor;
}