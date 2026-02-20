package CruiseControlSystem {
    private import ScalarValues::*;

    enum def SpeedUnit {
        enum KPH;
        enum MPH;
    }

    part def Speedometer {
        attribute currentSpeed : Real;
        attribute unit : SpeedUnit = SpeedUnit::KPH;
    }

    part def EngineSystem {
        port throttleControl : ThrottlePort;
    }

    port def ThrottlePort {
    }

    part def CruiseController {
        attribute setSpeed : Real;
        port speedInput : SpeedInputPort;
        port throttleOutput : ThrottlePort;
    }

    port def SpeedInputPort {
    }

    part def Driver {
        port setSpeedCommand : SetSpeedPort;
    }

    port def SetSpeedPort {
    }

    part vehicle {
        part speedometer : Speedometer;
        part engine : EngineSystem;
        part controller : CruiseController;
        part driver : Driver;

        connect driver.setSpeedCommand to controller;
        connect controller.speedInput to speedometer;
        connect controller.throttleOutput to engine.throttleControl;
    }
}