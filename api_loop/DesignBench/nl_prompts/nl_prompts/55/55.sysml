package 'VehicleMassCalculation' {
	private import ScalarValues::*;
	private import NumericalFunctions::*;
	part def VehicleComponent {
		attribute serialNumber : String;
		attribute simpleMass :> ISQ::mass;
		part subcomponents : VehicleComponent[*];
		attribute totalMass :> ISQ::mass default simpleMass + sum(subcomponents.totalMass);
		attribute threshold :> ISQ::mass;
		attribute filteredMass :> ISQ::mass default simpleMass + sum(subcomponents.totalMass.?{in m:>ISQ::mass; m >= threshold});
	}
	part def Vehicle :> VehicleComponent {
		attribute vin : String;
		part engine : VehicleComponent :>> subcomponents;
		part transmission : VehicleComponent :>> subcomponents;
		part otherComponents : VehicleComponent[*] :>> subcomponents;
	}
	part def VehicleMassManager {
		part vehicles : Vehicle[*];
		ref part focusedVehicle : Vehicle;
		attribute totalMassView :> ISQ::mass default focusedVehicle.totalMass;
		attribute filteredMassView :> ISQ::mass default focusedVehicle.filteredMass;
	}
}
