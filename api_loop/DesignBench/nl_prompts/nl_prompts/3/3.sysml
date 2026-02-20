package 'VehicleStructure' {
	private import ScalarValues::*;

	part def Vehicle {
		part eng : Engine[1];
		part transmission : Transmission[1];
		part wheel : Wheel[4];
	}

	part def VehicleComponent;

	part def Engine specializes VehicleComponent;
	part def Transmission specializes VehicleComponent;
	part def Wheel specializes VehicleComponent;
}
