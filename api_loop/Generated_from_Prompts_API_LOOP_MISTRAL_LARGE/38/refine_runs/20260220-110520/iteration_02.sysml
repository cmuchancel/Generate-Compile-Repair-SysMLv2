package VehicleControl {
    private import ScalarValues::*;

    enum def VehicleState {
        enum shutdown;
        enum start_up;
        enum running;
    }

    part def Vehicle {
        attribute currentState : VehicleState = VehicleState::shutdown;
        port startSignal : StartPort;
        port startedSignal : StartedPort;
        port shutdownSignal : ShutdownPort;
    }

    port def StartPort;
    port def StartedPort;
    port def ShutdownPort;

    part def User {
        port startCommand : StartPort;
        port shutdownCommand : ShutdownPort;
    }

    part def System {
        part vehicle : Vehicle;
        part user : User;

        connect user.startCommand to vehicle.startSignal;
        connect user.shutdownCommand to vehicle.shutdownSignal;
    }

    item def StartSignal;
    item def StartedSignal;
    item def ShutdownSignal;
}