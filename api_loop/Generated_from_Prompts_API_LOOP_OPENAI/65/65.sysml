package VehicleWeightControl {
  private import ScalarValues::*;

  enum def WeightStatus {
    enum OK;
    enum OVER_LIMIT;
  }

  part def Chassis {
    attribute weight : Real;
  }

  part def Engine {
    attribute weight : Real;
  }

  part def Transmission {
    attribute weight : Real;
  }

  part def VehicleWeightManager {
    part chassis : Chassis;
    part engine : Engine;
    part transmission : Transmission;
    attribute weightLimit : Real = 2500.0;
    attribute totalWeight : Real = chassis.weight + engine.weight + transmission.weight;
    attribute warning : Boolean = totalWeight > weightLimit;
    attribute status : WeightStatus;
  }

  part vehicleWeightManager : VehicleWeightManager;
}