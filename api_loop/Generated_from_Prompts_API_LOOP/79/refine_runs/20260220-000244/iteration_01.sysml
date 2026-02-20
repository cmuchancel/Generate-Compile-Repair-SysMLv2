package VehicleConfig {
  private import ScalarValues::*;

  part def Cylinder {
    attribute diameter : Real;
  }

  part def SmallCylinder specializes Cylinder {
    attribute diameter : Real = 70;
  }

  part def LargeCylinder specializes Cylinder {
    attribute diameter : Real = 100;
  }

  part def Engine {
    attribute cylinderCount : Integer;
  }

  part def FourCylinderEngine specializes Engine {
    attribute cylinderCount : Integer = 4;
    part cylinder1 : Cylinder;
    part cylinder2 : Cylinder;
    part cylinder3 : Cylinder;
    part cylinder4 : Cylinder;
  }

  part def SixCylinderEngine specializes Engine {
    attribute cylinderCount : Integer = 6;
    part cylinder1 : Cylinder;
    part cylinder2 : Cylinder;
    part cylinder3 : Cylinder;
    part cylinder4 : Cylinder;
    part cylinder5 : Cylinder;
    part cylinder6 : Cylinder;
  }

  part def Transmission;

  part def ManualTransmission specializes Transmission;

  part def AutomaticTransmission specializes Transmission;

  part def Vehicle {
    part engine : Engine;
    part transmission : Transmission;
  }

  part def VehicleFourCylinderManual specializes Vehicle {
    part engine : FourCylinderEngine;
    part transmission : ManualTransmission;
  }

  part def VehicleFourCylinderAutomatic specializes Vehicle {
    part engine : FourCylinderEngine;
    part transmission : AutomaticTransmission;
  }

  part def VehicleSixCylinderAutomatic specializes Vehicle {
    part engine : SixCylinderEngine;
    part transmission : AutomaticTransmission;
  }
}