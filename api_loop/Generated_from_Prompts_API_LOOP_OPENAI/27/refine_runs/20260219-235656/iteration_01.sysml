package BatteryChargingControl
{
  private import ScalarValues::*;
  
  enum def ChargeStatus {
    enum Charging;
    enum NotCharging;
  }
  
  item def BatteryLevelSignal {
    attribute level : Real;
  }
  
  item def ChargeCommand {
    attribute enable : Boolean;
  }
  
  port def LevelPort {
    attribute level : Real;
  }
  
  port def CommandPort {
    attribute enable : Boolean;
  }
  
  part def Battery {
    attribute capacityPercent : Real;
    port levelOut : LevelPort;
  }
  
  part def Charger {
    attribute isEnabled : Boolean;
    port cmdIn : CommandPort;
  }
  
  part def Controller {
    attribute targetFullPercent : Real = 100.0;
    attribute status : ChargeStatus;
    port levelIn : LevelPort;
    port cmdOut : CommandPort;
  }
  
  part def BatteryChargingSystem {
    part battery : Battery;
    part charger : Charger;
    part controller : Controller;
    connect battery.levelOut to controller.levelIn;
    connect controller.cmdOut to charger.cmdIn;
  }
}