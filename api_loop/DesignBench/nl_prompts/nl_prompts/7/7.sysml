package 'VehicleEngines' {
	private import ScalarValues::*;

	part def Vehicle;

	part def SmallVehicle specializes Vehicle {
		part engine : SmallEngine[1];
	}

	part def LargeVehicle specializes Vehicle {
		part engine : LargeEngine[1];
	}

	part def Engine {
		part cylinder : Cylinder[4..6];
	}

	part def SmallEngine specializes Engine {
		part cylinder redefines Engine::cylinder : Cylinder[4];
	}

	part def LargeEngine specializes Engine {
		part cylinder redefines Engine::cylinder : Cylinder[6];
	}

	part def Cylinder;
}
