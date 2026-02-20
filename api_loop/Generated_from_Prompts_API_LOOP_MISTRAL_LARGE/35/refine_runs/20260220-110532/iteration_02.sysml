package VehicleStartupShutdown {
    private import ScalarValues::*;

    enum def VehicleState {
        enum off;
        enum starting;
        enum on;
    }

    item def VehicleStartSignal;
    item def VehiclePowerOnSignal;
    item def VehicleShutdownSignal;

    port def StartPort {
        in start : VehicleStartSignal;
    }

    port def PowerPort {
        in power : VehiclePowerOnSignal;
    }

    port def ShutdownPort {
        in shutdown : VehicleShutdownSignal;
    }

    port def StatePort {
        out state : VehicleState;
    }

    part def VehicleController {
        attribute currentState : VehicleState = VehicleState::off;
        port startPort : StartPort;
        port powerPort : PowerPort;
        port shutdownPort : ShutdownPort;
        port statePort : StatePort;
    }

    part vehicleController : VehicleController;

    connect vehicleController.startPort to vehicleController.powerPort;
    connect vehicleController.powerPort to vehicleController.shutdownPort;
    connect vehicleController.shutdownPort to vehicleController.statePort;
}