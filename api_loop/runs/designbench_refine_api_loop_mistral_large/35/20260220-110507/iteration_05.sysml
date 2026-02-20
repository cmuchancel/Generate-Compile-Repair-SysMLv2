package VehicleStartupShutdown {
    private import ScalarValues::*;

    enum def VehicleState {
        enum off;
        enum starting;
        enum on;
    }

    part def Vehicle {
        attribute state : VehicleState = VehicleState::off;
    }

    port def StartSignalPort {
        attribute signal : StartSignal;
    }

    port def PowerOnSignalPort {
        attribute signal : PowerOnSignal;
    }

    port def ShutdownSignalPort {
        attribute signal : ShutdownSignal;
    }

    item def StartSignal;
    item def PowerOnSignal;
    item def ShutdownSignal;

    part vehicle : Vehicle {
        port startSignal : StartSignalPort;
        port powerOnSignal : PowerOnSignalPort;
        port shutdownSignal : ShutdownSignalPort;
    }

    part startSignalSource {
        port startOut : StartSignalPort;
    }

    part powerOnSignalSource {
        port powerOut : PowerOnSignalPort;
    }

    part shutdownSignalSource {
        port shutdownOut : ShutdownSignalPort;
    }

    connect startSignalSource.startOut to vehicle.startSignal;
    connect powerOnSignalSource.powerOut to vehicle.powerOnSignal;
    connect shutdownSignalSource.shutdownOut to vehicle.shutdownSignal;
}