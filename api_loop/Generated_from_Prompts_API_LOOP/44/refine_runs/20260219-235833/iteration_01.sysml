package CruiseControlSystemModel {
   private import ScalarValues::*;

   item def SetSpeedCommand;
   item def SpeedValue;
   item def FuelControlCommand;

   port def SetSpeedCmdPort {
      attribute cmd : SetSpeedCommand;
   }

   port def SpeedValuePort {
      attribute speed : SpeedValue;
   }

   port def FuelControlCmdPort {
      attribute fuelCmd : FuelControlCommand;
   }

   part def Driver {
      port setSpeedOut : SetSpeedCmdPort;
   }

   part def CruiseController {
      port setSpeedIn : SetSpeedCmdPort;
      port speedIn : SpeedValuePort;
      port fuelOut : FuelControlCmdPort;
      attribute desiredSpeed : Real;
   }

   part def Speedometer {
      port speedIn : SpeedValuePort;
      port speedOut : SpeedValuePort;
      attribute currentSpeed : Real;
   }

   part def Engine {
      port fuelIn : FuelControlCmdPort;
   }

   part def Vehicle {
      port speedOut : SpeedValuePort;
      attribute actualSpeed : Real;
   }

   part def CruiseControlSystem {
      part driver : Driver;
      part controller : CruiseController;
      part speedometer : Speedometer;
      part engine : Engine;
      part vehicle : Vehicle;

      connect driver.setSpeedOut to controller.setSpeedIn;
      connect vehicle.speedOut to speedometer.speedIn;
      connect speedometer.speedOut to controller.speedIn;
      connect controller.fuelOut to engine.fuelIn;
   }
}