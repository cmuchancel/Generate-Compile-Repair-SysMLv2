package VehicleStartupShutdown {
    private import ScalarValues::*;

    enum def VehicleState {
        enum off;
        enum starting;
        enum on;
    }

    part def Vehicle {
        attribute state : VehicleState = off;
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

    part startSignalSource :> StartSignal;
    part powerOnSignalSource :> PowerOnSignal;
    part shutdownSignalSource :> ShutdownSignal;

    connect vehicle.startSignal to startSignalSource;
    connect vehicle.powerOnSignal to powerOnSignalSource;
    connect vehicle.shutdownSignal to shutdownSignalSource;
}