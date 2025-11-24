package 'VehicleStructuralArchitecture' {
	private import ScalarValues::*;
	private import SI::*;
part def Wheel;
part def Axle;
part def FrontAxle :> Axle;
part def RearAxle :> Axle;
part def SteeringFrontAxle :> FrontAxle {
	attribute steeringAngle : Real;
}
part def AxleAssembly {
	part axle : Axle;
	part wheels : Wheel[2] ordered;
}
part def FrontAxleAssembly :> AxleAssembly {
	part axle : FrontAxle redefines AxleAssembly::axle;
}
part def RearAxleAssembly :> AxleAssembly {
	part axle : RearAxle redefines AxleAssembly::axle;
}
part def SteeringFrontAxleAssembly :> FrontAxleAssembly {
	part axle : SteeringFrontAxle redefines FrontAxleAssembly::axle;
}
part def NamedFrontAxleAssembly :> FrontAxleAssembly {
	part leftFrontWheel : Wheel redefines FrontAxleAssembly::wheels[1];
	part rightFrontWheel : Wheel redefines FrontAxleAssembly::wheels[2];
}
part def NamedRearAxleAssembly :> RearAxleAssembly {
	part leftRearWheel : Wheel redefines RearAxleAssembly::wheels[1];
	part rightRearWheel : Wheel redefines RearAxleAssembly::wheels[2];
}
part def Vehicle {
	attribute mass :> ISQ::mass;
	part frontAxleAssembly : FrontAxleAssembly;
	part rearAxleAssembly : RearAxleAssembly;
}
	part vehicle1750kg : Vehicle {
		attribute redefines mass = 1750[kg];
	}
	part vehicle2000kg : Vehicle {
		attribute redefines mass = 2000[kg];
	}
part def VehicleWithSteeringFrontAxle :> Vehicle {
	part frontAxleAssembly : SteeringFrontAxleAssembly redefines frontAxleAssembly;
}
part def VehicleWithNamedWheels :> Vehicle {
	part frontAxleAssembly : NamedFrontAxleAssembly redefines frontAxleAssembly;
	part rearAxleAssembly : NamedRearAxleAssembly redefines rearAxleAssembly;
}
}
