package PowertrainSystemModel {
  private import ScalarValues::*;

  item def FuelCommand {
    attribute level : Real;
  }

  item def StartStopCommand {
    attribute startCmd : Boolean;
    attribute stopCmd : Boolean;
  }

  item def Torque {
    attribute value : Real;
  }

  port def FuelCommandPort { }
  port def StartStopPort { }
  port def TorquePort { }

  part def DriverControls {
    port fuelOut : FuelCommandPort;
    port startStopOut : StartStopPort;
  }

  part def Engine {
    port fuelIn : FuelCommandPort;
    port startStopIn : StartStopPort;
    port torqueOut : TorquePort;
  }

  part def Transmission {
    port torqueIn : TorquePort;
    port torqueOut : TorquePort;
  }

  part def Driveshaft {
    port torqueIn : TorquePort;
    port torqueOut : TorquePort;
  }

  part def Differential {
    port torqueIn : TorquePort;
    port leftOut : TorquePort;
    port rightOut : TorquePort;
  }

  part def Wheel {
    port torqueIn : TorquePort;
  }

  part def PowertrainSystem {
    part driver : DriverControls;
    part engine : Engine;
    part transmission : Transmission;
    part driveshaft : Driveshaft;
    part differential : Differential;
    part leftWheel : Wheel;
    part rightWheel : Wheel;

    connect driver.fuelOut to engine.fuelIn;
    connect driver.startStopOut to engine.startStopIn;
    connect engine.torqueOut to transmission.torqueIn;
    connect transmission.torqueOut to driveshaft.torqueIn;
    connect driveshaft.torqueOut to differential.torqueIn;
    connect differential.leftOut to leftWheel.torqueIn;
    connect differential.rightOut to rightWheel.torqueIn;
  }
}