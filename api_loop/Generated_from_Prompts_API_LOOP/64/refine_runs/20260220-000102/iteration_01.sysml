package VehicleMassCompliance {
  private import ScalarValues::*;

  part def Chassis {
    attribute mass : Real;
  }

  part def Engine {
    attribute mass : Real;
  }

  part def Transmission {
    attribute mass : Real;
  }

  part def Vehicle {
    part chassis : Chassis;
    part engine : Engine;
    part transmission : Transmission;
    attribute maxMass : Real = 2500.0;
    attribute totalMass : Real;
    attribute compliant : Boolean;
  }

  part def MassComplianceSystem {
    part vehicle : Vehicle;
  }

  part massComplianceSystem : MassComplianceSystem;
}