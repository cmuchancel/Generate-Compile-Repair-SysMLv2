package VehicleModel {
  private import ScalarValues::*;

  part def Vehicle {
    attribute mass : Real = 1200;
    attribute length : Real = 4.82;
    part wheelAssembly[4] : WheelAssembly;
  }

  part def WheelAssembly {
    part wheel : Wheel;
    part brakeDisc : BrakeDisc;
    attribute brakeDiscDiameter : Real = brakeDisc.radius * 2;
    attribute brakeDiscFitsWheel : Boolean = (brakeDiscDiameter < wheel.outerDiameter);
    attribute brakeDiscFitsRim : Boolean = (brakeDiscDiameter < wheel.rim.diameterMM);
  }

  part def Wheel {
    part rim : Rim;
    part tire : Tire;
    attribute outerDiameter : Real = rim.diameterMM + 2 * tire.height;
  }

  part def Rim {
    attribute diameterInches : Real = 18;
    attribute diameterMM : Real = 457.2;
  }

  part def Tire {
    attribute width : Real = 245;
    attribute height : Real = 45;
    attribute treadDepth : Real = 6.0;
    attribute treadDepthOk : Boolean = (treadDepth >= 3.5);
  }

  part def BrakeDisc {
    attribute radius : Real = 95;
  }

  part vehicle : Vehicle;
}