package 'VehicleVariantConfigurations' {
	private import ScalarValues::*;
	private import SI::*;
	part def Tire {
		attribute mass :> ISQ::mass;
	}
	part def FrontAxle {
		attribute mass :> ISQ::mass;
		part tires : Tire[2] ordered;
	}
	part def SteeringFrontAxle :> FrontAxle {
		attribute steeringAngle : Real;
	}
	part def RearAxle {
		attribute mass :> ISQ::mass;
		part tires : Tire[2] ordered;
	}
	part def Vehicle {
		attribute mass :> ISQ::mass;
		part frontAxle : FrontAxle;
		part rearAxle : RearAxle;
	}
	part vehicleDefault : Vehicle {
		attribute redefines mass = 1750[kg];
	}
	part vehicle1_c1 : Vehicle {
		attribute redefines mass = 2000[kg];
		part steeringFrontAxle : SteeringFrontAxle redefines frontAxle {
			part leftFrontTire : Tire redefines tires[1];
			part rightFrontTire : Tire redefines tires[2];
		}
		part managedRearAxle : RearAxle redefines rearAxle {
			part leftRearTire : Tire redefines tires[1];
			part rightRearTire : Tire redefines tires[2];
		}
	}
}
