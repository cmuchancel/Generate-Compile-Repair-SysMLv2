package VehicleStateManagement {
    private import ScalarValues::*;

    enum def VehicleState {
        enum OFF;
        enum STARTING;
        enum ON;
    }

    enum def ControlSignal {
        enum VEHICLE_START_SIGNAL;
        enum VEHICLE_POWER_ON_SIGNAL;
        enum VEHICLE_SHUTDOWN_SIGNAL;
    }

    part def VehicleController {
        attribute currentState : VehicleState = VehicleState::OFF;
        attribute receivedSignal : ControlSignal;
        attribute initialState : VehicleState = VehicleState::OFF;
        attribute offState : VehicleState = VehicleState::OFF;
        attribute startingState : VehicleState = VehicleState::STARTING;
        attribute onState : VehicleState = VehicleState::ON;
    }

    part def SignalSource {
        attribute startSignal : ControlSignal = ControlSignal::VEHICLE_START_SIGNAL;
        attribute powerOnSignal : ControlSignal = ControlSignal::VEHICLE_POWER_ON_SIGNAL;
        attribute shutdownSignal : ControlSignal = ControlSignal::VEHICLE_SHUTDOWN_SIGNAL;
    }

    port def VehicleControlPort {
        attribute signal : ControlSignal;
    }

    port def VehicleStatePort {
        attribute state : VehicleState;
    }

    part def Vehicle {
        attribute vehicleState : VehicleState = VehicleState::OFF;
        port controlInput : VehicleControlPort;
        port stateOutput : VehicleStatePort;
    }

    part def VehicleSystem {
        part vehicle : Vehicle;
        part controller : VehicleController;
        part signalSource : SignalSource;

        port systemControlPort : VehicleControlPort;
        port systemStatePort : VehicleStatePort;

        connect systemControlPort to vehicle::controlInput;
        connect vehicle::stateOutput to systemStatePort;
    }

    part vehicleSystem : VehicleSystem;
}