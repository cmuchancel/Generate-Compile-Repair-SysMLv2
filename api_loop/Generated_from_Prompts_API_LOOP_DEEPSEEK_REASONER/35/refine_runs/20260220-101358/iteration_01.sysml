package VehicleStateManager {
    private import ScalarValues::*;

    enum def VehicleStateEnum {
        enum off;
        enum starting;
        enum on;
    }

    part def VehicleController {
        attribute currentState : VehicleStateEnum = VehicleStateEnum::off;
        attribute lastSignal : String;
        port inControlSignal : in VehicleControlSignal;
    }

    item def VehicleControlSignal {
        attribute signalType : String;
    }

    port def VehicleControlSignal in;

    part def SignalGenerator {
        port outControlSignal : out VehicleControlSignal;
    }

    part vehicleController : VehicleController;
    part signalGenerator : SignalGenerator;

    connect signalGenerator.outControlSignal to vehicleController.inControlSignal;
}