package VehicleSystemModel {
  private import ScalarValues::*;

  part def Wheel {
  }

  part def FrontAxle {
    attribute mass : Real;
    attribute steeringAngle : Real;
  }

  part def RearAxle {
    attribute mass : Real;
  }

  part def FrontAxleAssembly {
    attribute mass : Real = 150;
    part frontLeftWheel : Wheel;
    part frontRightWheel : Wheel;
    part frontAxle : FrontAxle;
  }

  part def RearAxleAssembly {
    attribute mass : Real = 250;
    part rearLeftWheel : Wheel;
    part rearRightWheel : Wheel;
    part rearAxle : RearAxle;
  }

  part def Car {
    attribute totalMass : Real = 2500;
    part frontAxleAssembly : FrontAxleAssembly;
    part rearAxleAssembly : RearAxleAssembly;
  }

  part car : Car;
}