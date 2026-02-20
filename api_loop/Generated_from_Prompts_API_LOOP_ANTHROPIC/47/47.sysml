package CruiseControlSystem {
    private import ScalarValues::*;

    enum def SpeedCommandStatus {
        enum PENDING;
        enum ACTIVE;
        enum INACTIVE;
    }

    enum def ThrottleStatus {
        enum INCREASE;
        enum DECREASE;
        enum HOLD;
    }

    item def SpeedCommand {
        attribute desiredSpeed : Real;
        attribute status : SpeedCommandStatus;
    }

    item def SpeedReading {
        attribute currentSpeed : Real;
    }

    item def ThrottleCommand {
        attribute throttleLevel : Real;
        attribute status : ThrottleStatus;
    }

    port def SpeedCommandPort {
        attribute desiredSpeed : Real;
    }

    port def SpeedReadingPort {
        attribute currentSpeed : Real;
    }

    port def ThrottleCommandPort {
        attribute throttleLevel : Real;
    }

    part def Driver {
        attribute driverName : String;
        port speedCommandOut : SpeedCommandPort;
    }

    part def Speedometer {
        attribute sensorAccuracy : Real;
        port speedReadingOut : SpeedReadingPort;
    }

    part def CruiseController {
        attribute setSpeed : Real;
        attribute currentSpeed : Real;
        attribute throttleAdjustment : Real;
        port speedCommandIn : SpeedCommandPort;
        port speedReadingIn : SpeedReadingPort;
        port throttleCommandOut : ThrottleCommandPort;
    }

    part def Engine {
        attribute throttleLevel : Real;
        attribute engineRPM : Real;
        port throttleCommandIn : ThrottleCommandPort;
    }

    part def Vehicle {
        attribute vehicleSpeed : Real;
        attribute cruiseEnabled : Boolean;

        part driver : Driver;
        part speedometer : Speedometer;
        part cruiseController : CruiseController;
        part engine : Engine;

        connect driver.speedCommandOut to cruiseController.speedCommandIn;
        connect speedometer.speedReadingOut to cruiseController.speedReadingIn;
        connect cruiseController.throttleCommandOut to engine.throttleCommandIn;
    }
}