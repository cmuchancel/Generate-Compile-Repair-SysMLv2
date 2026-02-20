package VehicleMassModel {
  private import ScalarValues::*;

  part def Component {
    attribute id : Integer;
    attribute mass : Real;
  }

  part def CompositeComponent : Component {
    attribute totalMass : Real;
    part subcomponents : Component[*];
  }

  part def Engine : Component {
    attribute mass : Real = 150.0;
  }

  part def Transmission : Component {
    attribute mass : Real = 90.0;
  }

  part def Body : Component {
    attribute mass : Real = 700.0;
  }

  part def Wheel : Component {
    attribute mass : Real = 25.0;
  }

  part def Vehicle : CompositeComponent {
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