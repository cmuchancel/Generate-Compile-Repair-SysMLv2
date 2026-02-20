package CruiseControlSystem {
    private import ScalarValues::*;

    enum def SpeedUnit {
        enum MPH;
        enum KPH;
    }

    part def Driver {
        port setSpeedPort;
    }

    part def Speedometer {
        port speedFeedbackPort;
        attribute currentSpeed : Real;
        attribute unit : SpeedUnit = SpeedUnit::MPH;
    }

    part def CruiseController {
        port speedCommandPort;
        port throttleControlPort;
        attribute setSpeed : Real;
        attribute isActive : Boolean = false;
        attribute throttleOutput : Real = 0.0;
    }

    part def Engine {
        port throttleInputPort;
    }

    part def Vehicle {
        part driver : Driver;
        part speedometer : Speedometer;
        part cruiseController : CruiseController;
        part engine : Engine;

        connect driver.setSpeedPort to cruiseController.speedCommandPort;
        connect speedometer.speedFeedbackPort to cruiseController.speedCommandPort;
        connect cruiseController.throttleControlPort to engine.throttleInputPort;
    }

    item def SpeedCommand {
        attribute desiredSpeed : Real;
        attribute unit : SpeedUnit = SpeedUnit::MPH;
    }

    item def ThrottleCommand {
        attribute throttleValue : Real;
    }
}