package VehicleStateManagement {
  private import ScalarValues::*;

  enum def VehicleState {
    enum off;
    enum starting;
    enum on;
  }

  part def VehicleController {
    attribute state : VehicleState = off;
    attribute vehicleStartSignal : Boolean = false;
    attribute vehiclePowerOnSignal : Boolean = false;
    attribute vehicleShutdownSignal : Boolean = false;
  }

  part vehicle : VehicleController;
}