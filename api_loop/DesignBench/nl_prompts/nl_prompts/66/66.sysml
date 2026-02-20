package 'VehicleMassAndDynamics' {
	private import ISQ::*;
	private import ScalarValues::*;
	private import SI::kg;
	part def VehicleComponent {
		attribute mass : MassValue;
	}
	part def VehicleAssembly {
		attribute massLimit : MassValue = 2500[kg];
		part chassis : VehicleComponent;
		part engine : VehicleComponent;
		part transmission : VehicleComponent;
		attribute totalMass : MassValue default chassis.mass + engine.mass + transmission.mass;
		attribute withinLimit : Boolean default totalMass <= massLimit;
	}
	calc def FinalVelocityFromForce {
		in initialVelocity : SpeedValue;
		in appliedForce : ForceValue;
		in vehicleMass : MassValue;
		in timeInterval : TimeValue;
		return : SpeedValue = initialVelocity + appliedForce * timeInterval / vehicleMass;
	}
	part def DynamicConstraintSet {
		attribute initialVelocity : SpeedValue;
		attribute finalVelocity : SpeedValue;
		attribute appliedForce : ForceValue;
		attribute timeInterval : TimeValue;
		attribute vehicleMass : MassValue;
		attribute massPositive : Boolean default vehicleMass > 0[kg];
		attribute impulseConsistency : Boolean default finalVelocity == FinalVelocityFromForce(
			initialVelocity,
			appliedForce,
			vehicleMass,
			timeInterval
		);
	}
	part def VehicleDynamicsVerifier {
		part assembly : VehicleAssembly;
		part dynamicConstraints : DynamicConstraintSet {
			attribute :>> vehicleMass default assembly.totalMass;
		}
		attribute satisfiesAllConstraints : Boolean default
			assembly.withinLimit and
			dynamicConstraints.massPositive and
			dynamicConstraints.impulseConsistency;
	}
}
