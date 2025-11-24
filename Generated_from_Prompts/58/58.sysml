package 'VehicleDynamicsCalculations' {
	private import ScalarValues::*;
	
	calc def EffectivePower {
		in wheelPower : Real;
		in aerodynamicDragCoefficient : Real;
		in frictionCoefficient : Real;
		in mass : Real;
		in speed : Real;
		return power : Real = wheelPower - aerodynamicDragCoefficient * speed - frictionCoefficient * mass;
	}
	
	calc def VehicleAcceleration {
		in effectivePower : Real;
		in mass : Real;
		in speed : Real;
		return acceleration : Real = effectivePower / (mass * speed);
	}
	
	calc def VehicleSpeed {
		in acceleration : Real;
		in initialSpeed : Real;
		in timeElapsed : Real;
		return speed : Real = initialSpeed + acceleration * timeElapsed;
	}
	
	calc def VehiclePosition {
		in speed : Real;
		in timeElapsed : Real;
		return displacement : Real = speed * timeElapsed;
	}
}
