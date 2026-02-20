package 'VehicleClassification' {
	private import ScalarValues::*;

	part def Vehicle;

	part def ManualVehicle specializes Vehicle {
		ref part driver : Personnel[1];
	}

	part def PoweredVehicle specializes Vehicle {
		part powerSource : Engine;
	}

	part def ManualPoweredVehicle specializes Vehicle {
		ref part driver : Personnel[1];
		part powerSource : Engine;
	}

	part def Personnel;
	part def Engine;
}
