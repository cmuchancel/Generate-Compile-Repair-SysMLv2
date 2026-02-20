package FuelEconomyModel {
  private import ScalarValues::*;

  enum def DrivingCondition {
    enum urban;
    enum highway;
  }

  part def FuelEconomyRequirement {
    attribute minUrbanMPG : Real = 25.0;
    attribute minHighwayMPG : Real = 30.0;
    attribute gallonVolumeCubicInches : Real = 231.0;
    attribute testLoadPounds : Real = 1000.0;
  }

  part def VehiclePowertrain {
    attribute efficiency : Real;
  }

  part def Vehicle {
    attribute baseMassPounds : Real;
    part powertrain : VehiclePowertrain;
  }

  part def FuelEconomyTest {
    attribute condition : DrivingCondition;
    attribute loadPounds : Real;
    attribute measuredMPG : Real;
  }

  part def FuelEconomySystem {
    part vehicle : Vehicle;
    part requirement : FuelEconomyRequirement;
    part urbanTest : FuelEconomyTest;
    part highwayTest : FuelEconomyTest;
  }
}