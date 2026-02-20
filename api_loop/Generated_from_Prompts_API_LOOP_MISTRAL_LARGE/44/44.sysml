package CruiseControlSystem {
    private import ScalarValues::*;

    enum def ActivationStatus {
        enum active;
        enum inactive;
    }

    part def Driver {
        attribute desiredSpeed : Real;
        port setSpeedPort : SetSpeedPort;
    }

    part def CruiseController {
        attribute currentSpeed : Real;
        attribute setSpeed : Real;
        attribute status : ActivationStatus = ActivationStatus::inactive;
        port speedInputPort : SpeedInputPort;
        port fuelControlPort : FuelControlPort;
    }

    part def Speedometer {
        attribute actualSpeed : Real;
        port speedOutputPort : SpeedOutputPort;
    }

    part def Engine {
        port fuelInputPort : FuelInputPort;
    }

    part def Vehicle {
        part driver : Driver;
        part cruiseController : CruiseController;
        part speedometer : Speedometer;
        part engine : Engine;
    }

    port def SetSpeedPort;
    port def SpeedInputPort;
    port def SpeedOutputPort;
    port def FuelControlPort;
    port def FuelInputPort;

    item def SpeedCommand {
        attribute speed : Real;
    }

    item def FuelCommand {
        attribute fuelRate : Real;
    }

    part vehicle : Vehicle;

    connect vehicle.driver.setSpeedPort to vehicle.cruiseController.speedInputPort;
    connect vehicle.speedometer.speedOutputPort to vehicle.cruiseController.speedInputPort;
    connect vehicle.cruiseController.fuelControlPort to vehicle.engine.fuelInputPort;
}