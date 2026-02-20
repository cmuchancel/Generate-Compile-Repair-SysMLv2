package CruiseControlSystem {
    private import ScalarValues::*;

    enum def SpeedUnit {
        enum kmph;
        enum mph;
    }

    part def SpeedSensor {
        attribute currentSpeed : Real;
        attribute unit : SpeedUnit = SpeedUnit::kmph;
        port speedOut : SpeedDataPort;
    }

    part def CruiseControlInterface {
        port setSpeedIn : SetSpeedPort;
        port speedDisplayOut : SpeedDisplayPort;
    }

    part def CruiseController {
        port setSpeedIn : SetSpeedPort;
        port actualSpeedIn : SpeedDataPort;
        port fuelControlOut : FuelControlPort;
    }

    part def Engine {
        port fuelControlIn : FuelControlPort;
    }

    port def SetSpeedPort {
        attribute targetSpeed : Real;
        attribute unit : SpeedUnit = SpeedUnit::kmph;
    }

    port def SpeedDataPort {
        attribute speedValue : Real;
        attribute unit : SpeedUnit = SpeedUnit::kmph;
    }

    port def FuelControlPort {
        attribute fuelCommand : Real;
    }

    port def SpeedDisplayPort {
        attribute displayedSpeed : Real;
        attribute unit : SpeedUnit = SpeedUnit::kmph;
    }

    part vehicle {
        part speedSensor : SpeedSensor;
        part cruiseInterface : CruiseControlInterface;
        part cruiseController : CruiseController;
        part engine : Engine;

        connect cruiseInterface.setSpeedIn to cruiseController.setSpeedIn;
        connect speedSensor.speedOut to cruiseController.actualSpeedIn;
        connect cruiseController.fuelControlOut to engine.fuelControlIn;
    }
}