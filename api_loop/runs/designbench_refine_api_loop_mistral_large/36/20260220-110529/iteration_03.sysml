package VehicleOperationalStatus {
    private import ScalarValues::*;

    enum def VehicleState {
        enum off;
        enum starting;
        enum on;
    }

    port def VehicleStartSignalPort {
        attribute signal : VehicleStartSignal;
    }

    port def VehicleOnSignalPort {
        attribute signal : VehicleOnSignal;
    }

    port def VehicleOffSignalPort {
        attribute signal : VehicleOnSignal;
    }

    item def VehicleStartSignal;
    item def VehicleOnSignal;
    item def VehicleOffSignal;

    part def Vehicle {
        attribute state : VehicleState = off;
        port startSignal : VehicleStartSignalPort;
        port onSignal : VehicleOnSignalPort;
        port offSignal : VehicleOffSignalPort;
    }

    part vehicle : Vehicle;

    part startSignalSource {
        port out : VehicleStartSignalPort;
    }

    part onSignalSource {
        port out : VehicleOnSignalPort;
    }

    part offSignalSource {
        port out : VehicleOffSignalPort;
    }

    connect vehicle.startSignal to startSignalSource.out;
    connect vehicle.onSignal to onSignalSource.out;
    connect vehicle.offSignal to offSignalSource.out;
}