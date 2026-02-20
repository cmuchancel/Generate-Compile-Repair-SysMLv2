package CruiseControlSystem {
    private import ScalarValues::*;

    item def SetSpeedCommand {
        attribute targetSpeed : Real;
    }

    item def SpeedReading {
        attribute currentSpeed : Real;
    }

    item def FuelControlCommand {
        attribute fuelAdjustment : Real;
    }

    port def CommandPort {
        attribute commandIn : SetSpeedCommand[0..1];
        attribute commandOut : SetSpeedCommand[0..1];
    }

    port def SensorPort {
        attribute readingOut : SpeedReading[0..1];
    }

    port def ControlPort {
        attribute controlIn : FuelControlCommand[0..1];
        attribute controlOut : FuelControlCommand[0..1];
    }

    part def Driver {
        port commandInterface : CommandPort;
    }

    part def CruiseController {
        port commandInterface : CommandPort;
        port sensorInterface : SensorPort;
        port controlInterface : ControlPort;
        attribute setSpeed : Real = 0.0;
        attribute actualSpeed : Real = 0.0;
    }

    part def SpeedSensor {
        port sensorInterface : SensorPort;
    }

    part def EngineController {
        port controlInterface : ControlPort;
        port engineInterface : ControlPort;
    }

    part def Engine {
        port engineInterface : ControlPort;
    }

    part CruiseControlSystem {
        part driver : Driver;
        part cruiseController : CruiseController;
        part speedSensor : SpeedSensor;
        part engineController : EngineController;
        part engine : Engine;

        connect driver.commandInterface to cruiseController.commandInterface;
        connect speedSensor.sensorInterface to cruiseController.sensorInterface;
        connect cruiseController.controlInterface to engineController.controlInterface;
        connect engineController.engineInterface to engine.engineInterface;
    }
}