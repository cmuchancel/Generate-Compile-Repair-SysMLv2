package VehicleOperationalStatus {
    private import ScalarValues::*;

    enum def VehicleState {
        enum off;
        enum starting;
        enum on;
    }

    item def VehicleStartSignal;
    item def VehicleOnSignal;
    item def VehicleOffSignal;

    port def VehicleStartSignalPort {
        in attribute signal : VehicleStartSignal;
    }

    port def VehicleOnSignalPort {
        in attribute signal : VehicleOnSignal;
    }

    port def VehicleOffSignalPort {
        in attribute signal : VehicleOffSignal;
    }

    part def Vehicle {
        attribute currentState : VehicleState = VehicleState::off;
        port startIn : VehicleStartSignalPort;
        port onIn : VehicleOnSignalPort;
        port offIn : VehicleOffSignalPort;
    }

    part vehicle : Vehicle;

    part def SignalSource {
        port startOut : VehicleStartSignalPort;
        port onOut : VehicleOnSignalPort;
        port offOut : VehicleOffSignalPort;
    }

    part signalSource : SignalSource;

    connect vehicle.startIn to signalSource.startOut;
    connect vehicle.onIn to signalSource.onOut;
    connect vehicle.offIn to signalSource.offOut;
}