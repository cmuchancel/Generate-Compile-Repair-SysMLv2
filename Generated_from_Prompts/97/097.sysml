package 'VehicleEngineConfiguration' {
	part def Vehicle {
		part engine : Engine;
	}
	part def Engine {
		part cylinders : Cylinder[4..6];
	}
	part def Cylinder;
	part def SmallVehicle :> Vehicle {
		part smallEngine : Engine redefines engine {
			part redefines cylinders[4];
		}
	}
	part def LargeVehicle :> Vehicle {
		part largeEngine : Engine redefines engine {
			part redefines cylinders[6];
		}
	}
}
