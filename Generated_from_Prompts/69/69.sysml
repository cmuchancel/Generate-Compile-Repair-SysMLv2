package 'VehicleMassPower069' {
	private import ScalarValues::*;
	
	part vehicle {
		attribute dryMass : Real;
		attribute fuelFullMass : Real;
		attribute fullTankMass : Real = dryMass + fuelFullMass;
		attribute emptyTankMass : Real = dryMass;
		constraint fullTankLimit { constraint { fullTankMass <= 2000.0 } }
		constraint emptyTankLimit { constraint { emptyTankMass <= 1500.0 } }
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
	
	part transmission {
		port clutchPort : ClutchPort;
	}
	
	connect engine.clutchPort to transmission.clutchPort;
}
