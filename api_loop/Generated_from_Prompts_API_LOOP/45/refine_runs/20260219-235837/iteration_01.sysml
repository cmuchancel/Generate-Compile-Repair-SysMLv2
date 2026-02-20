package CruiseControlSystem {
   private import ScalarValues::*;

   item def SetSpeedCommand {
      attribute setSpeed : Real;
   }

   item def VehicleSpeed {
      attribute speed : Real;
   }

   item def ThrottleCommand {
      attribute throttle : Real;
   }

   port def SetSpeedPort {
      item SetSpeedCommand;
   }

   port def SpeedPort {
      item VehicleSpeed;
   }

   port def ThrottlePort {
      item ThrottleCommand;
   }

   part def Driver {
      port setSpeedOut : SetSpeedPort;
   }

   part def Speedometer {
      port speedOut : SpeedPort;
   }

   part def EngineSystem {
      port throttleIn : ThrottlePort;
   }

   part def CruiseController {
      port setSpeedIn : SetSpeedPort;
      port speedIn : SpeedPort;
      port throttleOut : ThrottlePort;
      attribute targetSpeed : Real;
      attribute currentSpeed : Real;
      attribute controlThrottle : Real;
   }

   part def CruiseControlSystem {
      part driver : Driver;
      part speedometer : Speedometer;
      part engine : EngineSystem;
      part controller : CruiseController;

      connect driver.setSpeedOut to controller.setSpeedIn;
      connect speedometer.speedOut to controller.speedIn;
      connect controller.throttleOut to engine.throttleIn;
   }

   part system : CruiseControlSystem;
}