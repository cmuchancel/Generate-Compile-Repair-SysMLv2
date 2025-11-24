package 'VehicleDynamicSimulation' {
	private import ScalarValues::*;
	part def RoadCondition {
		attribute slopeAngleDegrees : Real;
		attribute frictionCoefficient : Real;
	}
	part def VehicleStateSnapshot {
		attribute timestamp : String;
		attribute massKilograms : Real;
		attribute positionMeters : Real;
		attribute velocityMetersPerSecond : Real;
		attribute accelerationMetersPerSecondSquared : Real;
		part road : RoadCondition;
	}
	part def SimulationSeries {
		part snapshots : VehicleStateSnapshot[*];
	}
	attribute def StartDrivingSignal;
	attribute def ShutdownSignal;
	state def VehicleOperationStates {
		in trackedVehicle : VehicleStateSnapshot;
	}
	state vehicleOperation : VehicleOperationStates {
		in trackedVehicle : VehicleStateSnapshot;
		entry; then startUp;
		state startUp;
		accept StartDrivingSignal
			then driving;
		state driving;
		accept ShutdownSignal
			then shutdown;
		state shutdown;
	}
	part def SimulationEnvironment {
		part recordedSeries : SimulationSeries;
		state vehicleStateFlow :> vehicleOperation;
	}
	part spatiotemporalSimulation : SimulationEnvironment {
		part recordedSeries : SimulationSeries redefines recordedSeries {
			part snapshots[3] : VehicleStateSnapshot redefines snapshots {
				part initialSnapshot : VehicleStateSnapshot redefines snapshots[1] {
					attribute timestamp = "2025-11-24T00:00:00Z";
					attribute massKilograms = 1500.0;
					attribute positionMeters = 0.0;
					attribute velocityMetersPerSecond = 0.0;
					attribute accelerationMetersPerSecondSquared = 0.5;
					part road : RoadCondition redefines road {
						attribute slopeAngleDegrees = 2.0;
						attribute frictionCoefficient = 0.85;
					}
				}
				part drivingSnapshot : VehicleStateSnapshot redefines snapshots[2] {
					attribute timestamp = "2025-11-24T00:05:00Z";
					attribute massKilograms = 1500.0;
					attribute positionMeters = 500.0;
					attribute velocityMetersPerSecond = 20.0;
					attribute accelerationMetersPerSecondSquared = 0.2;
					part road : RoadCondition redefines road {
						attribute slopeAngleDegrees = 4.0;
						attribute frictionCoefficient = 0.80;
					}
				}
				part shutdownSnapshot : VehicleStateSnapshot redefines snapshots[3] {
					attribute timestamp = "2025-11-24T00:20:00Z";
					attribute massKilograms = 1500.0;
					attribute positionMeters = 1200.0;
					attribute velocityMetersPerSecond = 0.0;
					attribute accelerationMetersPerSecondSquared = -0.3;
					part road : RoadCondition redefines road {
						attribute slopeAngleDegrees = 1.0;
						attribute frictionCoefficient = 0.90;
					}
				}
			}
		}
		state vehicleStateFlow :> vehicleOperation redefines vehicleStateFlow {
			in trackedVehicle = recordedSeries.snapshots.initialSnapshot;
		}
	}
	requirement def 'SpatiotemporalSimulationRequirement' {
		text = "Support time-series snapshots that capture vehicle mass, position, velocity, acceleration, and road slope/friction for each recorded moment.";
	}
	requirement def 'StateTrackingRequirement' {
		text = "Track vehicle state transitions from start-up to driving to shutdown while considering environmental factors during the simulation.";
	}
}
