package VehicleStateManager {
    private import ScalarValues::*;

    enum def VehicleStateEnum {
        enum off;
        enum starting;
        enum on;
    }

    item def VehicleControlSignal {
        attribute signalType : String;
    }

    port def VehicleSignalPort;

    part def VehicleController {
        attribute currentState : VehicleStateEnum = VehicleStateEnum::off;
        attribute lastSignal : String;
        port signalInput : VehicleSignalPort;
    }

    part def SignalGenerator {
        port signalOutput : VehicleSignalPort;
    }

    part vehicleController : VehicleController;
    part signalGenerator : SignalGenerator;

    connect signalGenerator.signalOutput to vehicleController.signalInput;
}