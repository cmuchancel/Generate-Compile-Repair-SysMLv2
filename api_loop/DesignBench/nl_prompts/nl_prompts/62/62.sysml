package 'VehicleMassRegulation' {
	private import ScalarValues::*;
	private import ISQ::*;
	private import SI::kg;
	part def VehicleComponent {
		attribute mass :> mass;
	}
	part def RegulatedVehicle {
		attribute massLimit :> mass = 2500[kg];
		part chassis : VehicleComponent;
		part engine : VehicleComponent;
		part gearbox : VehicleComponent;
		attribute totalMass :> mass default chassis.mass + engine.mass + gearbox.mass;
		attribute overweightWarning : Boolean default totalMass > massLimit;
	}
	part def VehicleMassMonitor {
		ref part vehicle : RegulatedVehicle;
		attribute compliant : Boolean default vehicle.totalMass <= vehicle.massLimit;
	}
}
