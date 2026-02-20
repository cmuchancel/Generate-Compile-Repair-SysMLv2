package VehicleStateManagement {
    private import ScalarValues::*;
    enum def VehicleState {
        enum off;
        enum starting;
        enum on;
    }
    enum def VehicleControlSignal {
        enum VehicleStartSignal;
        enum VehicleOnSignal;
        enum VehicleOffSignal;
    }
    port def ControlSignalPort {
        attribute signal : VehicleControlSignal;
    }
    part def VehicleStatusManager {
        attribute currentState : VehicleState = off;
        attribute lastSignal : VehicleControlSignal;
        port controlIn : ControlSignalPort;
    }
    part vehicleSystem : VehicleStatusManager;
}