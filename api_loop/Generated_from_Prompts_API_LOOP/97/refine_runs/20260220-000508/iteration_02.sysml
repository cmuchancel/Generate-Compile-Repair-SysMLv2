package VehicleModel {
  private import ScalarValues::*;

  enum def VehicleSize { small; standardSize; large; }

  part def Engine {
    attribute cylinderCount : Integer;
  }

  part def Engine4 specializes Engine {
    attribute cylinderCount = 4;
  }

  part def Engine6 specializes Engine {
    attribute cylinderCount = 6;
  }

  part def Vehicle {
    attribute size : VehicleSize;
    part engine : Engine;
  }

  part def SmallVehicle specializes Vehicle {
    attribute size = small;
    part engine : Engine4;
  }

  part def LargeVehicle specializes Vehicle {
    attribute size = large;
    part engine : Engine6;
  }

  part def StandardVehicle4 specializes Vehicle {
    attribute size = standardSize;
    part engine : Engine4;
  }

  part def StandardVehicle6 specializes Vehicle {
    attribute size = standardSize;
    part engine : Engine6;
  }
}