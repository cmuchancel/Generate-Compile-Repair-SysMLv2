package CruiseControlSystemPkg {
   private import ScalarValues::*;

   item def SetSpeedCommand {
      attribute setSpeed : Real;
   }

   item def SpeedReading {
      attribute speed : Real;
   }

   item def FuelControlCommand {
      attribute fuelRate : Real;
   }

   port def SetSpeedPort {
      attribute cmd : SetSpeedCommand;
   }

   port def SpeedPort {
      attribute reading : SpeedReading;
   }

   port def FuelCommandPort {
      attribute fuelCmd : FuelControlCommand;
   }

   part def Driver {
      port setSpeedOut : SetSpeedPort;
      attribute desiredSpeed : Real;
   }

   part def CruiseController {
      port setSpeedIn : SetSpeedPort;
      port speedIn : SpeedPort;
      port fuelCmdOut : FuelCommandPort;
      attribute setSpeed : Real;
      attribute actualSpeed : Real;
   }

   part def SpeedSensor {
      port speedOut : SpeedPort;
      attribute measuredSpeed : Real;
   }

   part def EngineController {
      port fuelCmdIn : FuelCommandPort;
      port fuelCmdOut : FuelCommandPort;
   }

   part def Engine {
      port fuelCmdIn : FuelCommandPort;
      attribute fuelFlow : Real;
   }

   part def CruiseControlSystem {
      part driver : Driver;
      part cruiseController : CruiseController;
      part speedSensor : SpeedSensor;
      part engineController : EngineController;
      part engine : Engine;

      connect driver.setSpeedOut to cruiseController.setSpeedIn;
      connect speedSensor.speedOut to cruiseController.speedIn;
      connect cruiseController.fuelCmdOut to engineController.fuelCmdIn;
      connect engineController.fuelCmdOut to engine.fuelCmdIn;
   }
}