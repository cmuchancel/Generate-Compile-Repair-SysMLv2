package 'VehicleMassAndPower068' {
	private import ScalarValues::*;
	
	part def Vehicle {
		attribute curbMass : Real;
		attribute fuelMass : Real;
		attribute totalMass : Real = curbMass + fuelMass;
		attribute maxAllowedMass : Real;
		constraint massLimit { constraint { totalMass <= maxAllowedMass } }
		constraint fuelMassPositive { constraint { fuelMass > 0.0 } }
	}
	
	port def ClutchPort;
	
	action def GenerateTorque {
		in speed : Real;
		out torque : Real;
	}
	
	part def Engine {
		port clutchPort : ClutchPort;
		perform action generateTorque : GenerateTorque;
	}
	
	part def Transmission {
		port clutchPort : ClutchPort;
	}
	
	part vehicleSystem {
		part vehicle : Vehicle {
			attribute maxAllowedMass = 2500.0;
		}
		part engine : Engine;
		part transmission : Transmission;
		connect engine.clutchPort to transmission.clutchPort;
	}
}
