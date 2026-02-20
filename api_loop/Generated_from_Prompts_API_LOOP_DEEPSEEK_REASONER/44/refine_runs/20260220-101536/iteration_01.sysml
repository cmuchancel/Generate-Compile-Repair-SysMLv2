package CruiseControlSystem {
    private import ScalarValues::*;

    item def SpeedCommand {
        attribute targetSpeed : Real;
    }

    item def SpeedReading {
        attribute currentSpeed : Real;
    }

    item def FuelCommand {
        attribute fuelLevel : Real;
    }

    port def DriverPort {
        attribute command : SpeedCommand;
    }

    port def ControllerPort {
        attribute setSpeed : SpeedCommand;
        attribute actualSpeed : SpeedReading;
        attribute fuelOut : FuelCommand;
    }

    port def SpeedometerPort {
        attribute speedOut : SpeedReading;
    }

    port def EnginePort {
        attribute fuelIn : FuelCommand;
    }

    part def Driver {
        port out commandPort : DriverPort;
    }

    part def CruiseController {
        port in driverIn : DriverPort;
        port in speedIn : SpeedometerPort;
        port out engineOut : EnginePort;
        attribute targetSpeed : Real;
        attribute currentSpeed : Real;
    }

    part def Speedometer {
        port out speedPort : SpeedometerPort;
        attribute measuredSpeed : Real;
    }

    part def Engine {
        port in controlPort : EnginePort;
        attribute fuelInput : Real;
    }

    part def Vehicle {
        part controller : CruiseController;
        part speedometer : Speedometer;
        part engine : Engine;
        connect controller.engineOut to engine.controlPort;
        connect speedometer.speedPort to controller.speedIn;
    }

    part def CruiseControlSystem {
        part driver : Driver;
        part vehicle : Vehicle;
        connect driver.commandPort to vehicle.controller.driverIn;
    }
}