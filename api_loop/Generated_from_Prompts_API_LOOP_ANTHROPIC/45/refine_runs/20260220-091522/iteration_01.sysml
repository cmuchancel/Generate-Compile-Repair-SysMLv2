package CruiseControlSystem {
    private import ScalarValues::*;

    enum def SpeedCommandStatus {
        enum SET;
        enum CLEAR;
    }

    enum def ControlState {
        enum ACTIVE;
        enum INACTIVE;
    }

    item def SpeedCommand {
        attribute targetSpeed : Real;
        attribute status : SpeedCommandStatus;
    }

    item def SpeedReading {
        attribute currentSpeed : Real;
    }

    item def ThrottleCommand {
        attribute throttleLevel : Real;
        attribute speedError : Real;
    }

    port def DriverCommandPort {
        attribute targetSpeed : Real;
        attribute commandStatus : SpeedCommandStatus;
    }

    port def SpeedSensorPort {
        attribute measuredSpeed : Real;
    }

    port def ThrottleControlPort {
        attribute throttleLevel : Real;
        attribute speedError : Real;
    }

    part def Driver {
        attribute desiredSpeed : Real;
        attribute commandStatus : SpeedCommandStatus;
        port commandOut : DriverCommandPort;
    }

    part def Speedometer {
        attribute vehicleSpeed : Real;
        port speedOut : SpeedSensorPort;
    }

    part def CruiseController {
        attribute setSpeed : Real;
        attribute actualSpeed : Real;
        attribute speedError : Real;
        attribute throttleOutput : Real;
        attribute controlState : ControlState;
        port commandIn : DriverCommandPort;
        port speedIn : SpeedSensorPort;
        port throttleOut : ThrottleControlPort;
    }

    part def EngineSystem {
        attribute powerOutput : Real;
        attribute throttleInput : Real;
        port throttleIn : ThrottleControlPort;
    }

    part def Vehicle {
        attribute currentSpeed : Real;

        part driver : Driver;
        part speedometer : Speedometer;
        part cruiseController : CruiseController;
        part engineSystem : EngineSystem;

        connect driver.commandOut to cruiseController.commandIn;
        connect speedometer.speedOut to cruiseController.speedIn;
        connect cruiseController.throttleOut to engineSystem.throttleIn;
    }
}