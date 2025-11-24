package 'AutomotiveConfigurationSpace' {
	private import SI::*;
	abstract part def VehicleConfiguration {
		part engine : Engine;
		part transmission : Transmission;
		assert constraint configurationRule : EngineTransmissionConstraint {
			in engine = VehicleConfiguration::engine;
			in transmission = VehicleConfiguration::transmission;
		}
	}
	abstract part def Engine {
		part cylinders : Cylinder[4..6];
	}
	part def FourCylinderEngine :> Engine {
		part redefines cylinders[4];
	}
	part def SixCylinderEngine :> Engine {
		part redefines cylinders[6];
	}
	abstract part def Cylinder {
		attribute diameter : LengthValue;
	}
	part def Cylinder70mm :> Cylinder {
		attribute redefines diameter = 70 [mm];
	}
	part def Cylinder100mm :> Cylinder {
		attribute redefines diameter = 100 [mm];
	}
	abstract part def Transmission;
	part def ManualTransmission :> Transmission;
	part def AutomaticTransmission :> Transmission;
	constraint def EngineTransmissionConstraint {
		in engine : Engine;
		in transmission : Transmission;
		(engine istype FourCylinderEngine implies transmission istype ManualTransmission)
			and (engine istype SixCylinderEngine implies transmission istype AutomaticTransmission)
	}
}
