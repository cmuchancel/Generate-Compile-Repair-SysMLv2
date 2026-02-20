package VehicleStructuralArchitecture {
  private import ScalarValues::*;

  part def Wheel;
  part def Axle;
  part def FrontAxle : Axle;
  part def RearAxle : Axle;

  part def SteerableFrontAxle : FrontAxle {
    attribute steeringAngle : Real;
  }

  part def AxleAssembly {
    part axle : Axle;
    part leftWheel : Wheel;
    part rightWheel : Wheel;
  }

  part def FrontAxleAssembly : AxleAssembly {
    part axle : FrontAxle;
  }

  part def RearAxleAssembly : AxleAssembly {
    part axle : RearAxle;
  }

  part def SteerableFrontAxleAssembly : FrontAxleAssembly {
    part axle : SteerableFrontAxle;
  }

  part def Vehicle {
    attribute mass : Real;
    part frontAxleAssembly : FrontAxleAssembly;
    part rearAxleAssembly : RearAxleAssembly;
  }

  part def Vehicle1750kg : Vehicle {
    attribute mass : Real = 1750;
  }

  part def Vehicle2000kg : Vehicle {
    attribute mass : Real = 2000;
  }

  part def VehicleWithSteerableFrontAxle {
    attribute mass : Real;
    part frontAxleAssembly : SteerableFrontAxleAssembly;
    part rearAxleAssembly : RearAxleAssembly;
  }
}