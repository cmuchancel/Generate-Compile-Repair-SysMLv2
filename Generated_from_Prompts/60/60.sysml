package 'VehicleDynamicsSimulation' {
	private import ScalarValues::*;
	
	part def SimulationState {
		attribute speed : Real;
		attribute position : Real;
	}
	
	part def SimulationParameters {
		attribute vehicleMass : Real;
		attribute wheelPower : Real;
		attribute aerodynamicDragCoefficient : Real;
		attribute rollingResistanceCoefficient : Real;
		attribute timeStep : Real;
	}
	
	calc def AvailablePower {
		in wheelPower : Real;
		in aerodynamicDragCoefficient : Real;
		in rollingResistanceCoefficient : Real;
		in speed : Real;
		return power : Real = wheelPower - aerodynamicDragCoefficient * speed - rollingResistanceCoefficient * speed;
	}
	
	calc def Acceleration {
		in availablePower : Real;
		in vehicleMass : Real;
		in speed : Real;
		return acceleration : Real = availablePower / (vehicleMass * speed);
	}
	
	calc def UpdateState {
		in currentState : SimulationState;
		in acceleration : Real;
		in timeStep : Real;
		return newState : SimulationState {
			attribute speed = currentState.speed + acceleration * timeStep;
			attribute position = currentState.position + currentState.speed * timeStep;
		}
	}
	
	part simulation {
		part parameters : SimulationParameters {
			attribute vehicleMass = 1500.0;
			attribute wheelPower = 50000.0;
			attribute aerodynamicDragCoefficient = 0.3;
			attribute rollingResistanceCoefficient = 0.02;
			attribute timeStep = 1.0;
		}
		part currentState : SimulationState {
			attribute speed = 20.0;
			attribute position = 0.0;
		}
		attribute availablePower : Real = parameters.wheelPower - parameters.aerodynamicDragCoefficient * currentState.speed - parameters.rollingResistanceCoefficient * currentState.speed;
		attribute acceleration : Real = availablePower / (parameters.vehicleMass * currentState.speed);
		part updatedState : SimulationState {
			attribute speed = currentState.speed + acceleration * parameters.timeStep;
			attribute position = currentState.position + currentState.speed * parameters.timeStep;
		}
	}
}
