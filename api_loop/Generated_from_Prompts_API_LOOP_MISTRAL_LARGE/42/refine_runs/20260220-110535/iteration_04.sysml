package VehicleStartStopSystem {
    private import ScalarValues::*;

    enum def StartStopState {
        enum off;
        enum starting;
        enum operating;
    }

    part def VehicleController {
        port controllerSignalPort : ControllerSignalPort;
    }

    part def BrakePedal {
        attribute isDepressed : Boolean;
    }

    part def ParkingBrake {
        attribute isApplied : Boolean;
    }

    port def StartSignalPort;
    port def PowerOnSignalPort;
    port def StopSignalPort;
    port def ControllerSignalPort;

    item def StartSignal;
    item def PowerOnSignal;
    item def StopSignal;
    item def ControllerCommand;

    part def VehicleStartStop {
        attribute state : StartStopState = StartStopState::off;

        part brakePedal : BrakePedal;
        part parkingBrake : ParkingBrake;
        part controller : VehicleController;

        port startSignalPort : StartSignalPort;
        port powerOnSignalPort : PowerOnSignalPort;
        port stopSignalPort : StopSignalPort;
        port controllerSignalPort : ControllerSignalPort;
    }

    connect VehicleStartStop::startSignalPort to VehicleStartStop::controllerSignalPort;
    connect VehicleStartStop::powerOnSignalPort to VehicleStartStop::controllerSignalPort;
    connect VehicleStartStop::stopSignalPort to VehicleStartStop::controller::controllerSignalPort;
}