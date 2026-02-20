package 'VehicleResources' {
	private import ScalarValues::*;

	part def Vehicle {
		attribute quality : Real;
		ref part driver : Personnel;
		part fuelTank : FuelTank[1];
	}

	part def FuelTank {
		ref part storedFuel : Fuel;
	}

	part def Fuel;
	part def Personnel;
}
