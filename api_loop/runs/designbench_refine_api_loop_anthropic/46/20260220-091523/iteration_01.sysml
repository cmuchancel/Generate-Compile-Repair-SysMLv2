package CruiseControlSystem {
    private import ScalarValues::*;

    enum def SpeedCommandStatus {
        enum SET;
        enum CANCEL;
    }

    enum def ThrottleCommandStatus {
        enum INCREASE;
        enum DECREASE;
        enum HOLD;
    }

    item def SpeedCommand {
        attribute targetSpeed : Real;
        attribute status : SpeedCommandStatus;
    }

    item def SpeedReading {
        attribute currentSpeed : Real;
    }

    item def ThrottleCommand {
        attribute throttleAdjustment : ThrottleCommandStatus;
        attribute throttleValue : Real;
    }

    port def DriverCommandPort {
        out item speedCommand : SpeedCommand;
    }

    port def ControllerInputPort {
        in item speedCommand : SpeedCommand;
    }

    port def SpeedSensorRequestPort {
        out item speedRequest : SpeedReading;
    }

    port def SpeedSensorResponsePort {
        in item speedReading : SpeedReading;
    }

    port def ThrottleOutputPort {
        out item throttleCommand : ThrottleCommand;
    }

    port def EngineInputPort {
        in item throttleCommand : ThrottleCommand;
    }

    port def SpeedometerOutputPort {
        out item speedReading : SpeedReading;
    }

    port def SpeedometerInputPort {
        in item speedRequest : SpeedReading;
    }

    part def Driver {
        attribute driverName : String;
        port commandOut : DriverCommandPort;
    }

    part def CruiseController {
        attribute setSpeed : Real;
        attribute currentSpeed : Real;
        attribute speedError : Real;
        port commandIn : ControllerInputPort;
        port speedSensorRequest : SpeedSensorRequestPort;
        port speedSensorResponse : SpeedSensorResponsePort;
        port throttleOut : ThrottleOutputPort;
    }

    part def Speedometer {
        attribute measuredSpeed : Real;
        port sensorInput : SpeedometerInputPort;
        port sensorOutput : SpeedometerOutputPort;
    }

    part def Engine {
        attribute engineOutput : Real;
        attribute currentRPM : Real;
        port throttleIn : EngineInputPort;
    }

    part def CruiseControlSystemAssembly {
        part driver : Driver;
        part cruiseController : CruiseController;
        part speedometer : Speedometer;
        part engine : Engine;

        connect driver.commandOut to cruiseController.commandIn;
        connect cruiseController.speedSensorRequest to speedometer.sensorInput;
        connect speedometer.sensorOutput to cruiseController.speedSensorResponse;
        connect cruiseController.throttleOut to engine.throttleIn;
    }
}