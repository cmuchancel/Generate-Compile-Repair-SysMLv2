package VehicleManagement {
    private import ScalarValues::*;

    enum def VehicleStatus {
        enum shutdown;
        enum startingUp;
        enum running;
    }

    enum def SignalType {
        enum startSignal;
        enum startedSignal;
        enum shutdownSignal;
    }

    item def VehicleSignal {
        attribute signalType : SignalType;
    }

    port def UserControlPort {
        attribute inputSignal : SignalType;
    }

    port def StatusPort {
        attribute currentStatus : VehicleStatus;
    }

    part def VehicleController {
        attribute vehicleStatus : VehicleStatus = VehicleStatus::shutdown;
        attribute lastSignalReceived : SignalType;
        port controlInput : UserControlPort;
        port statusOutput : StatusPort;
    }

    part def UserInterface {
        attribute userSignal : SignalType;
        port controlOutput : UserControlPort;
    }

    part def StatusMonitor {
        attribute monitoredStatus : VehicleStatus;
        port statusInput : StatusPort;
    }

    part def VehicleSystem {
        part controller : VehicleController;
        part userInterface : UserInterface;
        part statusMonitor : StatusMonitor;

        connect userInterface.controlOutput to controller.controlInput;
        connect controller.statusOutput to statusMonitor.statusInput;
    }
}