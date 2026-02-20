package CruiseControlSystem {
    private import ScalarValues::*;

    enum def SpeedCommandType {
        enum SET_SPEED;
        enum CANCEL;
    }

    enum def EngineCommandType {
        enum INCREASE_FUEL;
        enum DECREASE_FUEL;
        enum MAINTAIN_FUEL;
    }

    item def SpeedCommand {
        attribute commandType : SpeedCommandType;
        attribute desiredSpeed : Real;
    }

    item def SpeedInfo {
        attribute actualSpeed : Real;
    }

    item def FuelCommand {
        attribute commandType : EngineCommandType;
        attribute fuelLevel : Real;
    }

    port def DriverOutputPort {
        out item speedCommand : SpeedCommand;
    }

    port def ControllerInputPort {
        in item speedCommand : SpeedCommand;
    }

    port def SpeedometerOutputPort {
        out item speedInfo : SpeedInfo;
    }

    port def ControllerSpeedInputPort {
        in item speedInfo : SpeedInfo;
    }

    port def ControllerFuelOutputPort {
        out item fuelCommand : FuelCommand;
    }

    port def EngineInputPort {
        in item fuelCommand : FuelCommand;
    }

    port def VehicleSpeedOutputPort {
        out item speedInfo : SpeedInfo;
    }

    port def SpeedometerVehicleInputPort {
        in item speedInfo : SpeedInfo;
    }

    part def Driver {
        port driverOut : DriverOutputPort;
        attribute name : String;
    }

    part def CruiseController {
        port commandIn : ControllerInputPort;
        port speedIn : ControllerSpeedInputPort;
        port fuelOut : ControllerFuelOutputPort;
        attribute targetSpeed : Real;
    }

    part def Speedometer {
        port vehicleSpeedIn : SpeedometerVehicleInputPort;
        port speedOut : SpeedometerOutputPort;
        attribute measuredSpeed : Real;
    }

    part def Engine {
        port engineIn : EngineInputPort;
        attribute fuelLevel : Real;
    }

    part def Vehicle {
        port vehicleSpeedOut : VehicleSpeedOutputPort;
        attribute currentSpeed : Real;

        part controller : CruiseController;
        part speedometer : Speedometer;
        part engine : Engine;

        connect speedometer.vehicleSpeedIn to vehicleSpeedOut;
        connect speedometer.speedOut to controller.speedIn;
        connect controller.fuelOut to engine.engineIn;
    }

    part cruiseControlSystem {
        part driver : Driver;
        part vehicle : Vehicle;

        connect driver.driverOut to vehicle.controller.commandIn;
    }
}