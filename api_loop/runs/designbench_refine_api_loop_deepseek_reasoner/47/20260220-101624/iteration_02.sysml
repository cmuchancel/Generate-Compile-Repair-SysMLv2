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

    part def Driver {
        attribute setSpeed : DesiredSpeed;
    }

    part def Speedometer {
        attribute currentSpeed : VehicleSpeed;
    }

    part def CruiseController {
        attribute driverIn : DesiredSpeed;
        attribute sensorIn : VehicleSpeed;
        attribute actuatorOut : ThrottleCommand;
    }

    part def Engine {
        attribute throttleIn : ThrottleCommand;
    }

    part driver : Driver;
    part speedometer : Speedometer;
    part controller : CruiseController;
    part engine : Engine;

    connect driver.setSpeed to controller.driverIn;
    connect speedometer.currentSpeed to controller.sensorIn;
    connect controller.actuatorOut to engine.throttleIn;
}