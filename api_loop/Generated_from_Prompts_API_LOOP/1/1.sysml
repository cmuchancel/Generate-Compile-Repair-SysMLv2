package VehicleSystem {
  private import ScalarValues::*;

  enum def GearSetting {
    enum Park;
    enum Reverse;
    enum Neutral;
    enum Drive;
    enum Low;
  }

  part def VehicleStatus {
    attribute gearSetting : GearSetting;
    attribute acceleratorPedalPosition : Real;
  }

  part def Engine {
    attribute engineId : String;
    attribute engineType : String;
    attribute ratedPower : Real;
  }

  part def Driver {
    attribute driverId : String;
    attribute name : String;
    attribute licenseNumber : String;
  }

  part def Vehicle {
    attribute vehicleId : String;
    attribute mass : Real;
    part status : VehicleStatus;
    part engine : Engine;
    part driver : Driver;
  }
}