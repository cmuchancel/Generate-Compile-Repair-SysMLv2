package CruiseControlSystem {
    private import ScalarValues::*;

    item def DesiredSpeed {
        attribute value : Real;
    }

    item def VehicleSpeed {
        attribute value : Real;
    }

    item def ThrottleCommand {
        attribute level : Real;
    }

    port def DriverInterface {
        attribute setSpeed : DesiredSpeed;
    }

    port def SensorInterface {
        attribute currentSpeed : VehicleSpeed;
    }

    port def ActuatorInterface {
        attribute throttle : ThrottleCommand;
    }

    part def Driver {
        port interface : DriverInterface;
    }

    part def Speedometer {
        port sensorOut : SensorInterface;
    }

    part def CruiseController {
        port driverIn : DriverInterface;
        port sensorIn : SensorInterface;
        port actuatorOut : ActuatorInterface;
    }

    part def Engine {
        port actuatorIn : ActuatorInterface;
    }

    part system {
        part driver : Driver;
        part speedometer : Speedometer;
        part controller : CruiseController;
        part engine : Engine;

        connect driver.interface to controller.driverIn;
        connect speedometer.sensorOut to controller.sensorIn;
        connect controller.actuatorOut to engine.actuatorIn;
    }
}