package AutomotiveConfig {
  private import ScalarValues::*;

  enum def DiameterSpec {
    enum D70;
    enum D100;
  }

  part def Cylinder {
    attribute diameter : DiameterSpec;
  }

  part def Engine { }

  part def Engine4 : Engine {
    part cyl1 : Cylinder;
    part cyl2 : Cylinder;
    part cyl3 : Cylinder;
    part cyl4 : Cylinder;
  }

  part def Engine6 : Engine {
    part cyl1 : Cylinder;
    part cyl2 : Cylinder;
    part cyl3 : Cylinder;
    part cyl4 : Cylinder;
    part cyl5 : Cylinder;
    part cyl6 : Cylinder;
  }

  part def Transmission { }

  part def ManualTransmission : Transmission { }

  part def AutomaticTransmission : Transmission { }

  part def Vehicle {
    part engine : Engine;
    part transmission : Transmission;
  }

  part def Vehicle4Manual : Vehicle {
    part engine : Engine4;
    part transmission : ManualTransmission;
  }

  part def Vehicle6Auto : Vehicle {
    part engine : Engine6;
    part transmission : AutomaticTransmission;
  }
}