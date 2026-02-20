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

    part def Driver {
        attribute setSpeedCommand : SetSpeedCommand;
    }

    part def Speedometer {
        attribute currentSpeed : CurrentSpeed;
    }

    part def Engine {
        attribute throttleCommand : ThrottleCommand;
    }

    part def CruiseController {
        attribute setSpeedCommand : SetSpeedCommand;
        attribute currentSpeed : CurrentSpeed;
        attribute throttleCommand : ThrottleCommand;
        attribute setSpeed : Real;
        attribute currentSpeedValue : Real;
        attribute throttleAdjustment : Real;
    }

    part def CruiseControlSystem {
        part driver : Driver;
        part controller : CruiseController;
        part speedometer : Speedometer;
        part engine : Engine;

        connect driver to controller;
        connect speedometer to controller;
        connect controller to engine;
    }
}