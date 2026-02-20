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

    port def DriverInterface {
        attribute setSpeedCommand : SetSpeedCommand;
    }

    port def SensorInterface {
        attribute currentSpeed : CurrentSpeed;
    }

    port def ActuatorInterface {
        attribute throttleCommand : ThrottleCommand;
    }

    part def Driver {
        port interface : DriverInterface;
    }

    part def Speedometer {
        port output : SensorInterface;
    }

    part def Engine {
        port input : ActuatorInterface;
    }

    part def CruiseController {
        port driverInput : DriverInterface;
        port sensorInput : SensorInterface;
        port actuatorOutput : ActuatorInterface;
        
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