package 'VehicleMassBudgeting' {
	private import ScalarValues::*;
	private import NumericalFunctions::*;
	private import ISQ::*;
	private import SI::kg;
	part def VehicleComponentEntry {
		attribute mass :> ISQ::mass;
	}
	part def VehicleMassBudget {
		attribute limit :> ISQ::mass = 2500[kg];
		part chassis : VehicleComponentEntry;
		part engine : VehicleComponentEntry;
		part transmission : VehicleComponentEntry;
		part otherComponents : VehicleComponentEntry[*];
		attribute engineMassRecord :> ISQ::mass default engine.mass;
		attribute transmissionMassRecord :> ISQ::mass default transmission.mass;
		attribute totalMass :> ISQ::mass default chassis.mass + engine.mass + transmission.mass + sum(otherComponents.mass);
		attribute withinLimit : Boolean default totalMass <= limit;
	}
	part def MassBudgetController {
		ref part budget : VehicleMassBudget;
		attribute compliant : Boolean default budget.withinLimit;
	}
}
