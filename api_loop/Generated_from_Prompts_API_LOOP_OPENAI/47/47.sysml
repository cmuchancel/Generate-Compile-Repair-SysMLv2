package CruiseControlModel {
  private import ScalarValues::*;

  item def SpeedCommand;
  item def SpeedMeasurement;
  item def ThrottleCommand;

  port def SetSpeedPort {
    attribute setSpeed : Real;
  }

  port def SpeedPort {
    attribute currentSpeed : Real;
  }

  port def ThrottlePort {
    attribute throttle : Real;
  }

  part def Driver {
    port setSpeedOut : SetSpeedPort;
  }

  part def Speedometer {
    port speedOut : SpeedPort;
  }

  part def CruiseController {
    port setSpeedIn : SetSpeedPort;
    port speedIn : SpeedPort;
    port throttleOut : ThrottlePort;
  }

  part def Engine {
    port throttleIn : ThrottlePort;
  }

  part def CruiseControlSystem {
    part driver : Driver;
    part speedometer : Speedometer;
    part controller : CruiseController;
    part engine : Engine;

    connect driver.setSpeedOut to controller.setSpeedIn;
    connect speedometer.speedOut to controller.speedIn;
    connect controller.throttleOut to engine.throttleIn;
  }
}