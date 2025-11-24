package 'VehicleLongitudinalDynamics' {
	private import ScalarValues::*;
	
	part def SimulationInputs {
		attribute initialPosition : Real;
		attribute initialVelocity : Real;
		attribute mass : Real;
		attribute wheelPower : Real;
		attribute aerodynamicDragCoefficient : Real;
		attribute frictionCoefficient : Real;
		attribute timeInterval : Real;
	}
	
	calc def SolveAcceleration {
		in wheelPower : Real;
		in aerodynamicDragCoefficient : Real;
		in frictionCoefficient : Real;
		in mass : Real;
		in velocity : Real;
		return acceleration : Real = (wheelPower - aerodynamicDragCoefficient * velocity - frictionCoefficient * velocity) / (mass * velocity);
	}
	
	calc def SolveFinalVelocity {
		in acceleration : Real;
		in initialVelocity : Real;
		in timeInterval : Real;
		return velocity : Real = initialVelocity + acceleration * timeInterval;
	}
	
	calc def SolveFinalPosition {
		in initialPosition : Real;
		in initialVelocity : Real;
		in acceleration : Real;
		in timeInterval : Real;
		return position : Real = initialPosition + initialVelocity * timeInterval + 0.5 * acceleration * (timeInterval * timeInterval);
	}
	
	constraint def PowerBalanceConstraint {
		in wheelPower : Real;
		in aerodynamicDragCoefficient : Real;
		in frictionCoefficient : Real;
		in velocity : Real;
		constraint { wheelPower >= aerodynamicDragCoefficient * velocity + frictionCoefficient * velocity }
	}
	
	part simulationScenario {
		part inputs : SimulationInputs {
			attribute initialPosition = 0.0;
			attribute initialVelocity = 15.0;
			attribute mass = 1400.0;
			attribute wheelPower = 45000.0;
			attribute aerodynamicDragCoefficient = 0.32;
			attribute frictionCoefficient = 0.018;
			attribute timeInterval = 1.0;
		}
		attribute availableAcceleration : Real = (inputs.wheelPower - inputs.aerodynamicDragCoefficient * inputs.initialVelocity - inputs.frictionCoefficient * inputs.initialVelocity) / (inputs.mass * inputs.initialVelocity);
		attribute finalVelocity : Real = inputs.initialVelocity + availableAcceleration * inputs.timeInterval;
		attribute finalPosition : Real = inputs.initialPosition + inputs.initialVelocity * inputs.timeInterval + 0.5 * availableAcceleration * (inputs.timeInterval * inputs.timeInterval);
		assert constraint balance : PowerBalanceConstraint {
			in wheelPower = inputs.wheelPower;
			in aerodynamicDragCoefficient = inputs.aerodynamicDragCoefficient;
			in frictionCoefficient = inputs.frictionCoefficient;
			in velocity = inputs.initialVelocity;
		}
	}
}
