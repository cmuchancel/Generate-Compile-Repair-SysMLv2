package 'VehicleMassPower071' {
	private import ScalarValues::*;
	
	part vehicle {
		attribute curbMass : Real;
		attribute fuelMass : Real;
		attribute fuelFullMass : Real;
		attribute totalMass : Real = curbMass + fuelMass;
		constraint fullTankLimit {
			constraint { curbMass + fuelFullMass <= 2000.0 }
		}
		constraint emptyTankLimit {
			constraint { curbMass <= 1500.0 }
		}
	}
	
	port def ClutchPort;
	
	action def GenerateTorque {
		in speed : Real;
		out torque : Real;
	}
	
	part engine {
		port clutchPort : ClutchPort;
		perform action generateTorque : GenerateTorque;
	}
	
	part gearbox {
		port clutchPort : ClutchPort;
	}
	
	connect engine.clutchPort to gearbox.clutchPort;
}
