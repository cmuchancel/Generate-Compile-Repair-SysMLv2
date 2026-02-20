package VehicleStartupShutdown {
    private import ScalarValues::*;

    enum def VehicleState {
        enum off;
        enum starting;
        enum on;
    }

    part def Vehicle {
        attribute state : VehicleState = VehicleState::off;
        port startSignal : StartSignalPort;
        port powerOnSignal : PowerOnSignalPort;
        port shutdownSignal : ShutdownSignalPort;
    }

    port def StartSignalPort;
    port def PowerOnSignalPort;
    port def ShutdownSignalPort;

    item def StartSignal;
    item def PowerOnSignal;
    item def ShutdownSignal;

    part vehicle : Vehicle;

    connect vehicle.startSignal to startSignalSource;
    connect vehicle.powerOnSignal to powerOnSignalSource;
    connect vehicle.shutdownSignal to shutdownSignalSource;

    part startSignalSource;
    part powerOnSignalSource;
    part shutdownSignalSource;
}