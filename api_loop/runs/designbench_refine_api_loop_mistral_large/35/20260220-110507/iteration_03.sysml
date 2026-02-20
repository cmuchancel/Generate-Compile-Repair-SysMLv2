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

    part vehicle : Vehicle;

    part startSignalSource {
        port out : StartSignalPort;
    }

    part powerOnSignalSource {
        port out : PowerOnSignalPort;
    }

    part shutdownSignalSource {
        port out : ShutdownSignalPort;
    }

    connect startSignalSource.out to vehicle.startSignal;
    connect powerOnSignalSource.out to vehicle.powerOnSignal;
    connect shutdownSignalSource.out to vehicle.shutdownSignal;
}