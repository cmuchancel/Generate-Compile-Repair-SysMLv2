package VehicleModel {
  private import ScalarValues::*;
  
  enum def GearPosition {
    Neutral,
    Second
  }
  
  part def Driver;
  
  part def Engine;
  
  part def VehicleState {
    attribute mass : Real;
    attribute gear : GearPosition;
    attribute throttle : Real;
    part driver : Driver;
  }
  
  part def Vehicle {
    attribute mass : Real;
    attribute gear : GearPosition;
    attribute throttle : Real;
    part engine : Engine;
    part currentDriver : Driver;
    part state1 : VehicleState;
    part state2 : VehicleState;
  }
  
  part Alice : Driver;
  part Bob : Driver;
  
  part MyVehicle : Vehicle {
    attribute mass = 2000;
    attribute gear = GearPosition::Neutral;
    attribute throttle = 0;
    part engine;
    part currentDriver : Driver = Alice;
    part state1 : VehicleState {
      attribute mass = 2000;
      attribute gear = GearPosition::Neutral;
      attribute throttle = 0;
      part driver : Driver = Alice;
    }
    part state2 : VehicleState {
      attribute mass = 1500;
      attribute gear = GearPosition::Second;
      attribute throttle = 0.5;
      part driver : Driver = Bob;
    }
  }
}