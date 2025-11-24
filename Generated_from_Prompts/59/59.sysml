package 'VehicleStraightLineSimulation' {
	private import ScalarValues::*;
	
	calc def AvailableDrivingPower {
		in wheelPower : Real;
		in aerodynamicDragCoefficient : Real;
		in rollingResistanceCoefficient : Real;
		in speed : Real;
		return power : Real = wheelPower - aerodynamicDragCoefficient * speed - rollingResistanceCoefficient * speed;
	}
	
	calc def ComputeAcceleration {
		in availablePower : Real;
		in mass : Real;
		in speed : Real;
		return acceleration : Real = availablePower / (mass * speed);
	}
	
	calc def UpdateSpeed {
		in acceleration : Real;
		in initialSpeed : Real;
		in timeStep : Real;
		return newSpeed : Real = initialSpeed + acceleration * timeStep;
	}
	
	calc def UpdatePosition {
		in initialPosition : Real;
		in speed : Real;
		in timeStep : Real;
		return newPosition : Real = initialPosition + speed * timeStep;
	}
}
