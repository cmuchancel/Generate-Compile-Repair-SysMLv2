package VehicleModel {
  private import ScalarValues::*;

  enum def VehicleSize { small; standard; large; }

  part def Engine {
    attribute cylinderCount : Integer;
  }

  part def Engine4 : Engine {
    attribute cylinderCount : Integer = 4;
  }

  part def Engine6 : Engine {
    attribute cylinderCount : Integer = 6;
  }

  part def Vehicle {
    attribute size : VehicleSize;
    part engine : Engine;
  }

  part def SmallVehicle : Vehicle {
    attribute size : VehicleSize = VehicleSize::small;
    part engine : Engine4;
  }

  part def LargeVehicle : Vehicle {
    attribute size : VehicleSize = VehicleSize::large;
    part engine : Engine6;
  }

  part def StandardVehicle4 : Vehicle {
    attribute size : VehicleSize = VehicleSize::standard;
    part engine : Engine4;
  }

  part def StandardVehicle6 : Vehicle {
    attribute size : VehicleSize = VehicleSize::standard;
    part engine : Engine6;
  }
}