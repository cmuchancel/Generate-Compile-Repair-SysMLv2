package 'VehicleMassCompliance062' {
	private import ScalarValues::*;
	
	attribute MAX_VEHICLE_MASS : Real = 2500.0;
	
	part vehicle {
		part chassis {
			attribute mass : Real;
		}
		part engine {
			attribute mass : Real;
		}
		part gearbox {
			attribute mass : Real;
		}
		attribute totalMass : Real = chassis.mass + engine.mass + gearbox.mass;
		attribute overweightWarning : Boolean = totalMass > MAX_VEHICLE_MASS;
		constraint massLimit {
			constraint { totalMass <= MAX_VEHICLE_MASS }
		}
	}
}
