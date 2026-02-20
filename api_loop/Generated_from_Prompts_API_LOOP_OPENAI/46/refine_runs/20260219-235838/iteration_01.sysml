package CruiseControlSystem {
    private import ScalarValues::*;

    item def SpeedCommand {
        attribute targetSpeed : Real;
    }

    item def SpeedValue {
        attribute speed : Real;
    }

    item def ThrottleCommand {
        attribute throttle : Real;
    }

    port def SetSpeedOutPort {
        item speedCommand : SpeedCommand;
    }

    port def SetSpeedInPort {
        item speedCommand : SpeedCommand;
    }

    port def SpeedOutPort {
        item speedValue : SpeedValue;
    }

    port def SpeedInPort {
        item speedValue : SpeedValue;
    }

    port def ThrottleOutPort {
        item throttleCommand : ThrottleCommand;
    }

    port def ThrottleInPort {
        item throttleCommand : ThrottleCommand;
    }

    part def Driver {
        port setSpeedOut : SetSpeedOutPort;
        attribute desiredSpeed : Real;
    }

    part def CruiseController {
        port setSpeedIn : SetSpeedInPort;
        port speedIn : SpeedInPort;
        port throttleOut : ThrottleOutPort;
        attribute setSpeed : Real;
        attribute currentSpeed : Real;
        attribute speedError : Real;
    }

    part def Speedometer {
        port speedIn : SpeedInPort;
        port speedOut : SpeedOutPort;
        attribute measuredSpeed : Real;
    }

    part def Engine {
        port throttleIn : ThrottleInPort;
        port speedOut : SpeedOutPort;
        attribute outputSpeed : Real;
    }

    part def CruiseControlSystem {
        part driver : Driver;
        part controller : CruiseController;
        part speedometer : Speedometer;
        part engine : Engine;

        connect driver.setSpeedOut to controller.setSpeedIn;
        connect speedometer.speedOut to controller.speedIn;
        connect controller.throttleOut to engine.throttleIn;
        connect engine.speedOut to speedometer.speedIn;
    }

    part system : CruiseControlSystem;
}