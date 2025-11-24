package 'VehicleProductLineVariants' {
	private import SI::*;
	abstract part def Vehicle {
		part engine : Engine;
		part transmission : Transmission;
		assert constraint combinationRule : EngineTransmissionRule {
			in engine = Vehicle::engine;
			in transmission = Vehicle::transmission;
		}
	}
	abstract part def Engine {
		part cylinder : Cylinder[4..6];
	}
	part def FourCylinderEngine :> Engine {
		part redefines cylinder[4];
	}
	part def SixCylinderEngine :> Engine {
		part redefines cylinder[6];
	}
	abstract part def Cylinder {
		attribute diameter : LengthValue;
	}
	part def SmallDiameterCylinder :> Cylinder {
		attribute redefines diameter = 70 [mm];
	}
	part def LargeDiameterCylinder :> Cylinder {
		attribute redefines diameter = 100 [mm];
	}
	abstract part def Transmission;
	part def ManualTransmission :> Transmission;
	part def AutomaticTransmission :> Transmission;
	constraint def EngineTransmissionRule {
		in engine : Engine;
		in transmission : Transmission;
		(transmission istype ManualTransmission implies engine istype FourCylinderEngine)
			and (engine istype SixCylinderEngine implies transmission istype AutomaticTransmission)
	}
	part def FourCylinderManualVehicle :> Vehicle {
		part eng : FourCylinderEngine redefines engine;
		part gearbox : ManualTransmission redefines transmission;
	}
	part def SixCylinderAutomaticVehicle :> Vehicle {
		part eng : SixCylinderEngine redefines engine;
		part gearbox : AutomaticTransmission redefines transmission;
	}
}
