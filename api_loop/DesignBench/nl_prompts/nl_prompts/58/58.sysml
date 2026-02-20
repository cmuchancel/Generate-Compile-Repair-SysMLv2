package 'VehicleDynamicsPerformance' {
	private import ISQ::*;
	private import ScalarValues::Real;
	calc def AvailablePower {
		in wheelPower : PowerValue;
		in aerodynamicDragCoefficient : Real;
		in frictionCoefficient : Real;
		in vehicleMass : MassValue;
		in speed : SpeedValue;
		attribute aerodynamicDrag = aerodynamicDragCoefficient * speed;
		attribute rollingLoss = frictionCoefficient * vehicleMass * speed;
		return : PowerValue = wheelPower - aerodynamicDrag - rollingLoss;
	}
	calc def AccelerationFromPower {
		in availablePower : PowerValue;
		in vehicleMass : MassValue;
		in currentSpeed : SpeedValue;
		return : AccelerationValue = availablePower / (vehicleMass * currentSpeed);
	}
	calc def SpeedProjection {
		in acceleration : AccelerationValue;
		in initialSpeed : SpeedValue;
		in elapsedTime : TimeValue;
		return : SpeedValue = initialSpeed + acceleration * elapsedTime;
	}
	calc def DisplacementProjection {
		in speed : SpeedValue;
		in elapsedTime : TimeValue;
		return : LengthValue = speed * elapsedTime;
	}
	constraint def VehicleDynamicsRelations {
		in wheelPower : PowerValue;
		in aerodynamicDragCoefficient : Real;
		in frictionCoefficient : Real;
		in vehicleMass : MassValue;
		in currentSpeed : SpeedValue;
		in elapsedTime : TimeValue;
		in resultingSpeed : SpeedValue;
		in resultingAcceleration : AccelerationValue;
		in resultingDisplacement : LengthValue;
		attribute effectivePower : PowerValue = AvailablePower(
			wheelPower,
			aerodynamicDragCoefficient,
			frictionCoefficient,
			vehicleMass,
			currentSpeed
		);
		resultingAcceleration == AccelerationFromPower(effectivePower, vehicleMass, currentSpeed) and
		resultingSpeed == SpeedProjection(resultingAcceleration, currentSpeed, elapsedTime) and
		resultingDisplacement == DisplacementProjection(resultingSpeed, elapsedTime)
	}
	action def VehicleDynamicsEvaluation {
		in wheelPower : PowerValue;
		in aerodynamicDragCoefficient : Real;
		in frictionCoefficient : Real;
		in vehicleMass : MassValue;
		in currentSpeed : SpeedValue;
		in elapsedTime : TimeValue;
		out resultingSpeed : SpeedValue;
		out resultingAcceleration : AccelerationValue;
		out resultingDisplacement : LengthValue;
	    assert constraint dynamics : VehicleDynamicsRelations {
			in wheelPower = wheelPower;
			in aerodynamicDragCoefficient = aerodynamicDragCoefficient;
			in frictionCoefficient = frictionCoefficient;
			in vehicleMass = vehicleMass;
			in currentSpeed = currentSpeed;
			in elapsedTime = elapsedTime;
			in resultingSpeed = resultingSpeed;
			in resultingAcceleration = resultingAcceleration;
			in resultingDisplacement = resultingDisplacement;
		}
	}
}
