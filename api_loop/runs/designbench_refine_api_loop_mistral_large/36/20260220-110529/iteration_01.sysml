package VehicleOperationalStatus {
    private import ScalarValues::*;

    enum def VehicleState {
        enum off;
        enum starting;
        enum on;
    }

    part def Vehicle {
        attribute state : VehicleState = VehicleState::off;
        port startSignal : VehicleStartSignalPort;
        port onSignal : VehicleOnSignalPort;
        port offSignal : VehicleOffSignalPort;
    }

    part def VehicleStartSignalPort;
    part def VehicleOnSignalPort;
    part def VehicleOffSignalPort;

    item def VehicleStartSignal;
    item def VehicleOnSignal;
    item def VehicleOffSignal;

    part vehicle : Vehicle;

    connect vehicle.startSignal to VehicleStartSignal;
    connect vehicle.onSignal to VehicleOnSignal;
    connect vehicle.offSignal to VehicleOffSignal;
}