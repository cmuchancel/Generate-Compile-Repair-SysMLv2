package 'LongitudinalDynamicsSolver' {
	private import ISQ::*;
	private import ScalarValues::Real;
	calc def EffectivePower {
		in wheelPower : PowerValue;
		in dragCoefficient : Real;
		in frictionCoefficient : Real;
		in vehicleMass : MassValue;
		in currentSpeed : SpeedValue;
		attribute aerodynamicLoss = dragCoefficient * currentSpeed;
		attribute rollingLoss = frictionCoefficient * vehicleMass * currentSpeed;
		return : PowerValue = wheelPower - aerodynamicLoss - rollingLoss;
	}
	calc def AccelerationFromPower {
		in effectivePower : PowerValue;
		in vehicleMass : MassValue;
		in speed : SpeedValue;
		return : AccelerationValue = effectivePower / (vehicleMass * speed);
	}
	calc def FinalVelocity {
		in initialVelocity : SpeedValue;
		in acceleration : AccelerationValue;
		in timeInterval : TimeValue;
		return : SpeedValue = initialVelocity + acceleration * timeInterval;
	}
	calc def FinalPosition {
		in initialPosition : LengthValue;
		in initialVelocity : SpeedValue;
		in finalVelocity : SpeedValue;
		in timeInterval : TimeValue;
		attribute meanVelocity : SpeedValue = (initialVelocity + finalVelocity) / 2;
		return : LengthValue = initialPosition + meanVelocity * timeInterval;
	}
	constraint def LongitudinalDynamicsEquations {
		in wheelPower : PowerValue;
		in dragCoefficient : Real;
		in frictionCoefficient : Real;
		in vehicleMass : MassValue;
		in initialPosition : LengthValue;
		in initialVelocity : SpeedValue;
		in timeInterval : TimeValue;
		in resultingAcceleration : AccelerationValue;
		in resultingVelocity : SpeedValue;
		in resultingPosition : LengthValue;
		attribute effectivePower : PowerValue = EffectivePower(
			wheelPower,
			dragCoefficient,
			frictionCoefficient,
			vehicleMass,
			initialVelocity
		);
		attribute acceleration : AccelerationValue = AccelerationFromPower(
			effectivePower,
			vehicleMass,
			initialVelocity
		);
		attribute velocity : SpeedValue = FinalVelocity(
			initialVelocity,
			acceleration,
			timeInterval
		);
		attribute position : LengthValue = FinalPosition(
			initialPosition,
			initialVelocity,
			velocity,
			timeInterval
		);
		resultingAcceleration == acceleration and
		resultingVelocity == velocity and
		resultingPosition == position
	}
	action def EvaluateLongitudinalDynamics {
		in wheelPower : PowerValue;
		in dragCoefficient : Real;
		in frictionCoefficient : Real;
		in vehicleMass : MassValue;
		in initialPosition : LengthValue;
		in initialVelocity : SpeedValue;
		in timeInterval : TimeValue;
		out resultingAcceleration : AccelerationValue;
		out resultingVelocity : SpeedValue;
		out resultingPosition : LengthValue;
	    assert constraint dynamics : LongitudinalDynamicsEquations {
			in wheelPower = wheelPower;
			in dragCoefficient = dragCoefficient;
			in frictionCoefficient = frictionCoefficient;
			in vehicleMass = vehicleMass;
			in initialPosition = initialPosition;
			in initialVelocity = initialVelocity;
			in timeInterval = timeInterval;
			in resultingAcceleration = resultingAcceleration;
			in resultingVelocity = resultingVelocity;
			in resultingPosition = resultingPosition;
		}
	}
}
