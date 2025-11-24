package 'VehicleSpecificationManagement' {
	private import ScalarValues::*;
	port def DriveInterface {
		attribute torque : Real;
	}
	connection def DriveConnection {
		end enginePort : DriveInterface;
		end transmissionPort : DriveInterface;
	}
	part def Engine {
		port driveOutput : DriveInterface;
		attribute torqueOutput : Real;
	}
	part def Transmission {
		port driveInput : DriveInterface;
	}
	part def Vehicle {
		attribute mass : Real;
		attribute massLimit : Real;
		attribute fuelState : String;
		attribute reliabilityScore : Real;
		part engine : Engine;
		part transmission : Transmission;
		connection driveLink : DriveConnection {
			end enginePort = engine.driveOutput;
			end transmissionPort = transmission.driveInput;
		}
	}
	part Vehicle2000kgSpecification : Vehicle {
		attribute redefines mass = 2000;
		attribute redefines massLimit = 2000;
		attribute redefines fuelState = "Full";
		attribute redefines reliabilityScore = 0.98;
		part engine : Engine redefines engine {
			attribute redefines torqueOutput = 450;
		}
	}
	part Vehicle2500kgSpecification : Vehicle {
		attribute redefines mass = 2500;
		attribute redefines massLimit = 2500;
		attribute redefines fuelState = "Empty";
		attribute redefines reliabilityScore = 0.95;
		part engine : Engine redefines engine {
			attribute redefines torqueOutput = 380;
		}
	}
}
