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
        in item start : VehicleStartSignal;
    }

    port def PowerPort {
        in item power : VehiclePowerOnSignal;
    }

    port def ShutdownPort {
        in item shutdown : VehicleShutdownSignal;
    }

    port def StatePort {
        out item state : VehicleState;
    }

    part def VehicleController {
        attribute currentState : VehicleState = VehicleState::off;
        port startPort : StartPort;
        port powerPort : PowerPort;
        port shutdownPort : ShutdownPort;
        port statePort : StatePort;
    }

    part vehicleController : VehicleController;

    connect vehicleController.startPort to vehicleController;
    connect vehicleController.powerPort to vehicleController;
    connect vehicleController.shutdownPort to vehicleController;
    connect vehicleController.statePort to vehicleController;
}