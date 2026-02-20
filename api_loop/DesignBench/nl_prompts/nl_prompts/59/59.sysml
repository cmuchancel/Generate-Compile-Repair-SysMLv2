package 'StraightLineDrivingDynamics' {
	private import ISQ::*;
	private import ScalarValues::Real;
	calc def AvailableDrivingPower {
		in wheelPower : PowerValue;
		in aerodynamicDragCoefficient : Real;
		in rollingResistanceCoefficient : Real;
		in vehicleMass : MassValue;
		in currentSpeed : SpeedValue;
		attribute aerodynamicDrag = aerodynamicDragCoefficient * currentSpeed;
		attribute rollingResistance = rollingResistanceCoefficient * vehicleMass * currentSpeed;
		return : PowerValue = wheelPower - aerodynamicDrag - rollingResistance;
	}
	calc def AccelerationFromPower {
		in availablePower : PowerValue;
		in vehicleMass : MassValue;
		in currentSpeed : SpeedValue;
		return : AccelerationValue = availablePower / (vehicleMass * currentSpeed);
	}
	calc def SpeedAfterStep {
		in initialSpeed : SpeedValue;
		in acceleration : AccelerationValue;
		in timeStep : TimeValue;
		return : SpeedValue = initialSpeed + acceleration * timeStep;
	}
	calc def PositionAfterStep {
		in initialPosition : LengthValue;
		in initialSpeed : SpeedValue;
		in resultingSpeed : SpeedValue;
		in timeStep : TimeValue;
		attribute averageSpeed : SpeedValue = (initialSpeed + resultingSpeed) / 2;
		return : LengthValue = initialPosition + averageSpeed * timeStep;
	}
	constraint def StraightLineStepRelations {
		in wheelPower : PowerValue;
		in aerodynamicDragCoefficient : Real;
		in rollingResistanceCoefficient : Real;
		in vehicleMass : MassValue;
		in initialSpeed : SpeedValue;
		in initialPosition : LengthValue;
		in timeStep : TimeValue;
		in resultingAcceleration : AccelerationValue;
		in resultingSpeed : SpeedValue;
		in resultingPosition : LengthValue;
		attribute availablePower : PowerValue = AvailableDrivingPower(
			wheelPower,
			aerodynamicDragCoefficient,
			rollingResistanceCoefficient,
			vehicleMass,
			initialSpeed
		);
		resultingAcceleration == AccelerationFromPower(availablePower, vehicleMass, initialSpeed) and
		resultingSpeed == SpeedAfterStep(initialSpeed, resultingAcceleration, timeStep) and
		resultingPosition == PositionAfterStep(initialPosition, initialSpeed, resultingSpeed, timeStep)
	}
	action def StraightLineDrivingStep {
		in wheelPower : PowerValue;
		in aerodynamicDragCoefficient : Real;
		in rollingResistanceCoefficient : Real;
		in vehicleMass : MassValue;
		in initialSpeed : SpeedValue;
		in initialPosition : LengthValue;
		in timeStep : TimeValue;
		out resultingAcceleration : AccelerationValue;
		out resultingSpeed : SpeedValue;
		out resultingPosition : LengthValue;
	    assert constraint dynamics : StraightLineStepRelations {
			in wheelPower = wheelPower;
			in aerodynamicDragCoefficient = aerodynamicDragCoefficient;
			in rollingResistanceCoefficient = rollingResistanceCoefficient;
			in vehicleMass = vehicleMass;
			in initialSpeed = initialSpeed;
			in initialPosition = initialPosition;
			in timeStep = timeStep;
			in resultingAcceleration = resultingAcceleration;
			in resultingSpeed = resultingSpeed;
			in resultingPosition = resultingPosition;
		}
	}
}
