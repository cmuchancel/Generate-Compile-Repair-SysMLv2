package VehicleStructuralArchitecture {
  private import ScalarValues::*;

  part def Wheel;
  part def Axle;
  part def FrontAxle specializes Axle;
  part def RearAxle specializes Axle;

  part def SteerableFrontAxle specializes FrontAxle {
    attribute steeringAngle : Real;
  }

  part def AxleAssembly {
    part axle : Axle;
    part leftWheel : Wheel;
    part rightWheel : Wheel;
  }

  part def FrontAxleAssembly specializes AxleAssembly {
    part axle : FrontAxle;
  }

  part def RearAxleAssembly specializes AxleAssembly {
    part axle : RearAxle;
  }

  part def SteerableFrontAxleAssembly specializes FrontAxleAssembly {
    part axle : SteerableFrontAxle;
  }

  part def Vehicle {
    attribute mass : Real;
    part frontAxleAssembly : FrontAxleAssembly;
    part rearAxleAssembly : RearAxleAssembly;
  }

  part def Vehicle1750kg specializes Vehicle {
    attribute mass : Real = 1750;
  }

  part def Vehicle2000kg specializes Vehicle {
    attribute mass : Real = 2000;
  }

  part def VehicleWithSteerableFrontAxle {
    attribute mass : Real;
    part frontAxleAssembly : SteerableFrontAxleAssembly;
    part rearAxleAssembly : RearAxleAssembly;
  }
}