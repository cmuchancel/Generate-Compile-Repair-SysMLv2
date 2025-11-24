package 'VehicleConstraints066' {
	private import ScalarValues::*;
	
	attribute VEHICLE_MASS_LIMIT : Real = 2500.0;
	
	part vehicle {
		part chassis { attribute mass : Real; }
		part engine { attribute mass : Real; }
		part transmission { attribute mass : Real; }
		attribute totalMass : Real = chassis.mass + engine.mass + transmission.mass;
		constraint totalMassLimit { constraint { totalMass <= VEHICLE_MASS_LIMIT } }
	}
	
	part dynamicScenario {
		attribute mass : Real;
		attribute initialVelocity : Real;
		attribute finalVelocity : Real;
		attribute appliedForce : Real;
		attribute timeDuration : Real;
		constraint massPositive { constraint { mass > 0.0 } }
		constraint momentumBalance {
			constraint { appliedForce * timeDuration == mass * (finalVelocity - initialVelocity) }
		}
	}
}
