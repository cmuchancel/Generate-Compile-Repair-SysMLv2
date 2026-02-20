package 'VehicleMotionSimulation31' {
	private import ScalarValues::*;

	part def VehicleMotionSimulation {
		part vehicle : Vehicle;
		part simulationSetup : SimulationSetup;
		part powerSequence : PowerInputSequence;
		part positionTrajectory : PositionTrajectory;
		attribute oneDimensionalPath : Boolean;
		attribute motionBehaviorAnalyzed : Boolean;
	}

	part def Vehicle {
		attribute massInput : MassValue;
	}

	part def SimulationSetup {
		attribute initialPosition : PositionValue;
		attribute initialVelocity : VelocityValue;
		attribute stepInterval : TimeIntervalValue;
	}

	part def PowerInputSequence {
		attribute powerInputs : PowerSeries;
	}

	part def PositionTrajectory {
		attribute stepPositions : PositionSeries;
	}

	attribute def MassValue {
		attribute mass : Real;
	}

	attribute def PositionValue {
		attribute position : Real;
	}

	attribute def VelocityValue {
		attribute velocity : Real;
	}

	attribute def TimeIntervalValue {
		attribute interval : Real;
	}

	attribute def PowerSeries {
		attribute powerPerStep : Sequence;
	}

	attribute def PositionSeries {
		attribute positionPerStep : Sequence;
	}

	attribute def Sequence;
}
