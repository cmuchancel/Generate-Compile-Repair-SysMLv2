package 'VehicleMassConstraint063' {
	private import ScalarValues::*;
	
	attribute MaxVehicleMass : Real = 2500.0;
	
	part vehicle {
		attribute chassisMassKilograms : Real;
		attribute engineMassKilograms : Real;
		attribute transmissionMassKilograms : Real;
		attribute totalMassKilograms : Real = chassisMassKilograms + engineMassKilograms + transmissionMassKilograms;
		attribute meetsMassRequirement : Boolean = totalMassKilograms <= MaxVehicleMass;
		constraint massRequirement {
			constraint { totalMassKilograms <= MaxVehicleMass }
		}
	}
}
