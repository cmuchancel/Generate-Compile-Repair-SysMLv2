package CruiseControlSystem {
    private import ScalarValues::*;

    enum def CommandType {
        enum SET_SPEED;
        enum FUEL_CONTROL;
    }

    enum def SensorStatus {
        enum ACTIVE;
        enum INACTIVE;
    }

    item def SpeedCommand {
        attribute targetSpeed : Real;
        attribute commandType : CommandType;
    }

    item def FuelControlCommand {
        attribute fuelAdjustment : Real;
        attribute commandType : CommandType;
    }

    item def SpeedReading {
        attribute currentSpeed : Real;
        attribute status : SensorStatus;
    }

    port def DriverInputPort {
        in item speedCmd : SpeedCommand;
    }

    port def DriverOutputPort {
        out item speedCmd : SpeedCommand;
    }

    port def SpeedSensorOutputPort {
        out item speedReading : SpeedReading;
    }

    port def SpeedSensorInputPort {
        in item speedReading : SpeedReading;
    }

    port def FuelCommandOutputPort {
        out item fuelCmd : FuelControlCommand;
    }

    port def FuelCommandInputPort {
        in item fuelCmd : FuelControlCommand;
    }

    part def Driver {
        attribute driverName : String;
        port driverOut : DriverOutputPort;
    }

    part def SpeedSensor {
        attribute sensorId : String;
        attribute measuredSpeed : Real;
        port sensorOut : SpeedSensorOutputPort;
    }

    part def CruiseController {
        attribute controllerId : String;
        attribute setSpeed : Real;
        attribute actualSpeed : Real;
        port cmdIn : DriverInputPort;
        port sensorIn : SpeedSensorInputPort;
        port fuelCmdOut : FuelCommandOutputPort;
    }

    part def EngineController {
        attribute engineControllerId : String;
        port fuelCmdIn : FuelCommandInputPort;
        port fuelCmdOut : FuelCommandOutputPort;
    }

    part def Engine {
        attribute engineId : String;
        attribute fuelFlow : Real;
        port fuelCmdIn : FuelCommandInputPort;
    }

    part cruiseControlSystem {
        part driver : Driver {
            attribute driverName = "Driver1";
            attribute setSpeed : Real;
        }

        part speedSensor : SpeedSensor {
            attribute sensorId = "SpeedSensor1";
            attribute measuredSpeed : Real = 0.0;
        }

        part cruiseController : CruiseController {
            attribute controllerId = "CruiseController1";
            attribute setSpeed : Real = 0.0;
            attribute actualSpeed : Real = 0.0;
        }

        part engineController : EngineController {
            attribute engineControllerId = "EngineController1";
        }

        part engine : Engine {
            attribute engineId = "Engine1";
            attribute fuelFlow : Real = 0.0;
        }

        connect driver.driverOut to cruiseController.cmdIn;
        connect speedSensor.sensorOut to cruiseController.sensorIn;
        connect cruiseController.fuelCmdOut to engineController.fuelCmdIn;
        connect engineController.fuelCmdOut to engine.fuelCmdIn;
    }
}