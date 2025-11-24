package 'VehicleLongitudinalDynamics' {
	private import ScalarValues::*;
	part def PowerSegment {
		attribute power : Real;
		attribute duration : Real;
	}
	part def VehicleDynamicsInput {
		attribute mass : Real;
		attribute initialPosition : Real;
		attribute initialVelocity : Real;
		attribute timeStep : Real;
		part powerSegments : PowerSegment[*] ordered;
	}
	calc def ComputeAcceleration {
		in power : Real;
		in mass : Real;
		return acceleration : Real;
	}
	calc def UpdateVelocity {
		in currentVelocity : Real;
		in acceleration : Real;
		in timeStep : Real;
		return velocity : Real;
	}
	calc def UpdatePosition {
		in currentPosition : Real;
		in velocity : Real;
		in timeStep : Real;
		return position : Real;
	}
	part def MotionSeries {
		attribute accelerationSeries : Real[*];
		attribute velocitySeries : Real[*];
		attribute positionSeries : Real[*];
	}
	part simulationCase {
		part input : VehicleDynamicsInput {
			attribute mass = 1800;
			attribute initialPosition = 0;
			attribute initialVelocity = 0;
			attribute timeStep = 0.5;
			part powerSegments {
				part segment1 : PowerSegment {
					attribute power = 160;
					attribute duration = 2.0;
				}
				part segment2 : PowerSegment {
					attribute power = 120;
					attribute duration = 2.0;
				}
			}
		}
		calc accelerationCalc : ComputeAcceleration {
			in power = input.powerSegments.segment1.power;
			in mass = input.mass;
		}
		calc velocityCalc : UpdateVelocity {
			in currentVelocity = input.initialVelocity;
			in acceleration = accelerationCalc.acceleration;
			in timeStep = input.timeStep;
		}
		calc positionCalc : UpdatePosition {
			in currentPosition = input.initialPosition;
			in velocity = velocityCalc.velocity;
			in timeStep = input.timeStep;
		}
		part motionSeries : MotionSeries {
			attribute accelerationSeries = (accelerationCalc.acceleration, accelerationCalc.acceleration * 0.75);
			attribute velocitySeries = (velocityCalc.velocity, velocityCalc.velocity + 4.0);
			attribute positionSeries = (positionCalc.position, positionCalc.position + 6.0);
		}
	}
}
