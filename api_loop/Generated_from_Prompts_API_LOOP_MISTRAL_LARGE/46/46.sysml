package CruiseControlSystem {
    private import ScalarValues::*;

    enum def SpeedUnit {
        enum kmph;
        enum mph;
    }

    part def Driver {
        port setSpeedPort : SetSpeedPort;
    }

    part def CruiseController {
        port setSpeedPort : SetSpeedPort;
        port currentSpeedPort : CurrentSpeedPort;
        port throttleControlPort : ThrottleControlPort;
    }

    part def Speedometer {
        port currentSpeedPort : CurrentSpeedPort;
        attribute currentSpeed : Real;
        attribute unit : SpeedUnit = SpeedUnit::kmph;
    }

    part def Engine {
        port throttleControlPort : ThrottleControlPort;
    }

    port def SetSpeedPort;
    port def CurrentSpeedPort;
    port def ThrottleControlPort;

    item def SpeedCommand {
        attribute targetSpeed : Real;
        attribute unit : SpeedUnit = SpeedUnit::kmph;
    }

    item def ThrottleCommand {
        attribute throttleValue : Real;
    }

    part vehicle {
        part driver : Driver;
        part cruiseController : CruiseController;
        part speedometer : Speedometer;
        part engine : Engine;

        connect driver.setSpeedPort to cruiseController.setSpeedPort;
        connect cruiseController.currentSpeedPort to speedometer.currentSpeedPort;
        connect cruiseController.throttleControlPort to engine.throttleControlPort;
    }
}