package 'VehicleWeightControl' {
	private import ScalarValues::*;
	private import ISQ::*;
	private import SI::kg;
	part def VehicleComponent {
		attribute weight :> ISQ::mass;
	}
	part def RegulatedAssembly {
		attribute weightLimit :> ISQ::mass = 2500[kg];
		part chassis : VehicleComponent;
		part engine : VehicleComponent;
		part transmission : VehicleComponent;
		attribute combinedWeight :> ISQ::mass default chassis.weight + engine.weight + transmission.weight;
		attribute warning : Boolean default combinedWeight > weightLimit;
	}
	part def WeightComplianceSystem {
		part assembly : RegulatedAssembly;
		attribute restrictionRequired : Boolean default assembly.warning;
	}
}
