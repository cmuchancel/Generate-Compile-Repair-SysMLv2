package VehicleOperationalStatus {
    private import ScalarValues::*;

    enum def VehicleState {
        enum off;
        enum starting;
        enum on;
    }

    part def VehicleStartSignalPort {
        attribute signal : VehicleStartSignal;
    }

    part def VehicleOnSignalPort {
        attribute signal : VehicleOnSignal;
    }

    part def VehicleOffSignalPort {
        attribute signal : VehicleOffSignal;
    }

    item def VehicleStartSignal;
    item def VehicleOnSignal;
    item def VehicleOffSignal;

    part def Vehicle {
        attribute state : VehicleState = VehicleState::off;
        port startSignal : VehicleStartSignalPort;
        port onSignal : VehicleOnSignalPort;
        port offSignal : VehicleOffSignalPort;
    }

    part vehicle : Vehicle;

    connect vehicle.startSignal to VehicleStartSignalPort;
    connect vehicle.onSignal to VehicleOnSignalPort;
    connect vehicle.offSignal to VehicleOffSignalPort;
}