package AutomotiveCruiseControl {
    private import ScalarValues::*;

    item def SetSpeedCommand {
        attribute targetSpeed : Real;
    }

    item def CurrentSpeed {
        attribute speed : Real;
    }

    item def ThrottleCommand {
        attribute adjustment : Real;
    }

    port def DriverInterface;

    port def SensorInterface;

    port def ActuatorInterface;

    part def Driver {
        attribute interface : DriverInterface;
    }

    part def Speedometer {
        attribute output : SensorInterface;
    }

    part def Engine {
        attribute input : ActuatorInterface;
    }

    part def CruiseController {
        attribute driverInput : DriverInterface;
        attribute sensorInput : SensorInterface;
        attribute actuatorOutput : ActuatorInterface;
        attribute setSpeed : Real;
        attribute currentSpeed : Real;
        attribute throttleAdjustment : Real;
    }

    part def CruiseControlSystem {
        part driver : Driver;
        part controller : CruiseController;
        part speedometer : Speedometer;
        part engine : Engine;

        connect driver.interface to controller.driverInput;
        connect speedometer.output to controller.sensorInput;
        connect controller.actuatorOutput to engine.input;
    }
}