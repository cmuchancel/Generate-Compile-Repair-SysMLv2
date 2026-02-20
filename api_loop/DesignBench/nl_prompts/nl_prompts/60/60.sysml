package 'VehicleDynamicsSimulation' {
	private import ISQ::*;
	private import ScalarValues::Real;
	calc def EffectiveDrivingPower {
		in wheelPower : PowerValue;
		in aerodynamicDragCoefficient : Real;
		in rollingResistanceCoefficient : Real;
		in vehicleMass : MassValue;
		in speed : SpeedValue;
		attribute aerodynamicDrag = aerodynamicDragCoefficient * speed;
		attribute rollingResistance = rollingResistanceCoefficient * vehicleMass * speed;
		return : PowerValue = wheelPower - aerodynamicDrag - rollingResistance;
	}
	calc def AccelerationFromPower {
		in effectivePower : PowerValue;
		in vehicleMass : MassValue;
		in speed : SpeedValue;
		return : AccelerationValue = effectivePower / (vehicleMass * speed);
	}
	calc def SpeedUpdate {
		in speed : SpeedValue;
		in acceleration : AccelerationValue;
		in timeStep : TimeValue;
		return : SpeedValue = speed + acceleration * timeStep;
	}
	calc def PositionUpdate {
		in position : LengthValue;
		in initialSpeed : SpeedValue;
		in updatedSpeed : SpeedValue;
		in timeStep : TimeValue;
		attribute meanSpeed : SpeedValue = (initialSpeed + updatedSpeed) / 2;
		return : LengthValue = position + meanSpeed * timeStep;
	}
	part def DynamicsInputs {
		attribute wheelPower : PowerValue;
		attribute aerodynamicDragCoefficient : Real;
		attribute rollingResistanceCoefficient : Real;
		attribute vehicleMass : MassValue;
	}
	part def VehicleState {
		attribute speed : SpeedValue;
		attribute position : LengthValue;
		attribute acceleration : AccelerationValue;
		attribute timeStamp : TimeValue;
	}
	constraint def TimeStepDynamics {
		in inputs : DynamicsInputs;
		in enteringState : VehicleState;
		in timeStep : TimeValue;
		in resultingState : VehicleState;
		attribute effectivePower : PowerValue = EffectiveDrivingPower(
			inputs.wheelPower,
			inputs.aerodynamicDragCoefficient,
			inputs.rollingResistanceCoefficient,
			inputs.vehicleMass,
			enteringState.speed
		);
		attribute acceleration : AccelerationValue = AccelerationFromPower(
			effectivePower,
			inputs.vehicleMass,
			enteringState.speed
		);
		attribute updatedSpeed : SpeedValue = SpeedUpdate(
			enteringState.speed,
			acceleration,
			timeStep
		);
		attribute updatedPosition : LengthValue = PositionUpdate(
			enteringState.position,
			enteringState.speed,
			updatedSpeed,
			timeStep
		);
		resultingState.acceleration == acceleration and
		resultingState.speed == updatedSpeed and
		resultingState.position == updatedPosition and
		resultingState.timeStamp == enteringState.timeStamp + timeStep
	}
	action def SimulateTimeStep {
		in inputs : DynamicsInputs;
		in enteringState : VehicleState;
		in timeStep : TimeValue;
		out resultingState : VehicleState;
	    assert constraint dynamics : TimeStepDynamics {
			in inputs = inputs;
			in enteringState = enteringState;
			in timeStep = timeStep;
			in resultingState = resultingState;
		}
	}
	part def VehicleDynamicsModel {
		part inputs : DynamicsInputs;
		part initialState : VehicleState;
		part simulationSteps : SimulateTimeStep[*];
	}
}
