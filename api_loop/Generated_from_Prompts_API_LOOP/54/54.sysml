package VehicleMassModel {
  private import ScalarValues::*;

  part def Component {
    attribute id : Integer;
    attribute mass : Real;
  }

  part def CompositeComponent {
    attribute id : Integer;
    attribute mass : Real;
    attribute totalMass : Real;
    part subcomponents : Component[*];
  }

  part def Engine {
    attribute id : Integer;
    attribute mass : Real = 150.0;
  }

  part def Transmission {
    attribute id : Integer;
    attribute mass : Real = 90.0;
  }

  part def Body {
    attribute id : Integer;
    attribute mass : Real = 700.0;
  }

  part def Wheel {
    attribute id : Integer;
    attribute mass : Real = 25.0;
  }

  part def Vehicle {
    attribute id : Integer;
    attribute mass : Real;
    attribute totalMass : Real;
    part engine : Engine;
    part transmission : Transmission;
    part body : Body;
    part wheelFL : Wheel;
    part wheelFR : Wheel;
    part wheelRL : Wheel;
    part wheelRR : Wheel;
  }

  part vehicle1 : Vehicle;
}