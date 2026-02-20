package VehicleModel {
  private import ScalarValues::*;

  enum def VehicleSize {
    enum small;
    enum standardSize;
    enum large;
  }

  part def Engine {
    attribute cylinderCount : Integer;
  }

  part def Engine4 {
    attribute cylinderCount : Integer = 4;
  }

  part def Engine6 {
    attribute cylinderCount : Integer = 6;
  }

  part def Vehicle {
    attribute size : VehicleSize;
    part engine : Engine;
  }

  part def SmallVehicle {
    attribute size : VehicleSize = VehicleSize::small;
    part engine : Engine4;
  }

  part def LargeVehicle {
    attribute size : VehicleSize = VehicleSize::large;
    part engine : Engine6;
  }

  part def StandardVehicle4 {
    attribute size : VehicleSize = VehicleSize::standardSize;
    part engine : Engine4;
  }

  part def StandardVehicle6 {
    attribute size : VehicleSize = VehicleSize::standardSize;
    part engine : Engine6;
  }
}