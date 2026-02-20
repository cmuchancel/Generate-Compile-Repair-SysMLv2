package 'VehicleMassCompliance' {
	private import ScalarValues::*;
	private import ISQ::*;
	private import SI::kg;
	part def VehicleSection {
		attribute mass :> ISQ::mass;
	}
	part def Vehicle {
		attribute massLimit :> ISQ::mass = 2500[kg];
		part chassis : VehicleSection;
		part engine : VehicleSection;
		part transmission : VehicleSection;
		attribute totalMass :> ISQ::mass default chassis.mass + engine.mass + transmission.mass;
		attribute compliant : Boolean default totalMass <= massLimit;
	}
	part def VehicleMassComplianceSystem {
		part regulatedVehicle : Vehicle;
		attribute meetsRequirement : Boolean default regulatedVehicle.compliant;
	}
}
