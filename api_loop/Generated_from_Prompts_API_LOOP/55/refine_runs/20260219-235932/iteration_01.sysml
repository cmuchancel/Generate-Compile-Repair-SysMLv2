package VehicleMassManagement {
  private import ScalarValues::*;

  part def Component {
    attribute serialNumber : String;
    attribute ownMass : Real;
    attribute massThreshold : Real;
    attribute totalMass : Real;
    part subcomponents : Component[*];
  }

  part def Engine : Component {
  }

  part def Transmission : Component {
  }

  part def Vehicle {
    attribute vin : String;
    attribute ownMass : Real;
    attribute massThreshold : Real;
    attribute totalMass : Real;
    part engine : Engine;
    part transmission : Transmission;
    part components : Component[*];
  }

  part vehicle : Vehicle;
}