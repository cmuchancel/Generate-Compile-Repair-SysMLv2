package VehicleStartStopSystem {
    private import ScalarValues::*;

    enum def VehicleState {
        enum off;
        enum starting;
        enum started;
        enum shuttingDown;
    }

    part def VehicleController {
        attribute currentState : VehicleState = VehicleState::off;
    }

    part def BrakeSystem {
        attribute isDepressed : Boolean = false;
    }

    part def PowerSystem {
        attribute isPowerSupplied : Boolean = false;
    }

    part def ParkingBrake {
        attribute isApplied : Boolean = false;
    }

    part def StartStopSystem {
        attribute startSignalReceived : Boolean = false;
        attribute shutdownSignalReceived : Boolean = false;
        attribute vehicleOnSignalReceived : Boolean = false;

        part vehicleController : VehicleController;
        part brakeSystem : BrakeSystem;
        part powerSystem : PowerSystem;
        part parkingBrake : ParkingBrake;
    }

    port def ControlSignalPort {
        attribute signal : String;
    }

    part vehicleStartStop : StartStopSystem;

    connect vehicleStartStop.vehicleController to vehicleStartStop.brakeSystem;
    connect vehicleStartStop.vehicleController to vehicleStartStop.powerSystem;
    connect vehicleStartStop.vehicleController to vehicleStartStop.parkingBrake;
}