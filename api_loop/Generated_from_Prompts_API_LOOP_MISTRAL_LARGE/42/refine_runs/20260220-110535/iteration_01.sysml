package VehicleStartStopSystem {
    private import ScalarValues::*;

    enum def StartStopState {
        enum off;
        enum starting;
        enum operating;
    }

    part def VehicleController;

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

    part def VehicleStartStopSystem {
        attribute state : StartStopState = StartStopState::off;

        part brakePedal : BrakePedal;
        part parkingBrake : ParkingBrake;
        part controller : VehicleController;

        port startSignalPort : StartSignalPort;
        port powerOnSignalPort : PowerOnSignalPort;
        port stopSignalPort : StopSignalPort;
        port controllerSignalPort : ControllerSignalPort;
    }

    connect VehicleStartStopSystem::startSignalPort to VehicleStartStopSystem::controllerSignalPort;
    connect VehicleStartStopSystem::powerOnSignalPort to VehicleStartStopSystem::controllerSignalPort;
    connect VehicleStartStopSystem::stopSignalPort to VehicleStartStopSystem::parkingBrake;
}