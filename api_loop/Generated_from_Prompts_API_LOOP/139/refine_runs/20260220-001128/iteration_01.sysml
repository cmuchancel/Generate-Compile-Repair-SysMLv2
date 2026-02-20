package VehicleModel {
  private import ScalarValues::*;

  part def Vehicle {
    attribute mass : Real = 1200[kg];
    attribute length : Real = 4.82[m];
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
    attribute diameterInches : Real = 18[in];
    attribute diameterMM : Real = 457.2[mm];
  }

  part def Tire {
    attribute width : Real = 245[mm];
    attribute height : Real = 45[mm];
    attribute treadDepth : Real = 6.0[mm];
    attribute treadDepthOk : Boolean = (treadDepth >= 3.5[mm]);
  }

  part def BrakeDisc {
    attribute radius : Real = 95[mm];
  }

  part vehicle : Vehicle;
}