package VehicleDynamics {
    private import ScalarValues::*;

    enum def DrivingMode {
        enum STRAIGHT_LINE;
        enum CRUISE;
        enum ACCELERATION;
    }

    part def VehicleParameters {
        attribute totalMass : Real;
        attribute aeroDragCoefficient : Real;
        attribute rollingResistanceCoefficient : Real;
        attribute frontalArea : Real;
        attribute airDensity : Real;
    }

    part def InitialConditions {
        attribute initialSpeed : Real;
        attribute initialPosition : Real;
        attribute timeStep : Real;
    }

    part def PowerInput {
        attribute wheelEndOutputPower : Real;
    }

    part def DragForceCalculator {
        attribute currentSpeed : Real;
        attribute aeroDragCoefficient : Real;
        attribute frontalArea : Real;
        attribute airDensity : Real;
        attribute aeroDragForce : Real;
    }

    part def RollingResistanceCalculator {
        attribute totalMass : Real;
        attribute rollingResistanceCoefficient : Real;
        attribute gravitationalAcceleration : Real = 9.81;
        attribute rollingResistanceForce : Real;
    }

    part def AvailablePowerCalculator {
        attribute wheelEndOutputPower : Real;
        attribute aeroDragForce : Real;
        attribute rollingResistanceForce : Real;
        attribute currentSpeed : Real;
        attribute powerLostToDrag : Real;
        attribute powerLostToRolling : Real;
        attribute availableDrivingPower : Real;
    }

    part def AccelerationCalculator {
        attribute availableDrivingPower : Real;
        attribute currentSpeed : Real;
        attribute totalMass : Real;
        attribute netForce : Real;
        attribute acceleration : Real;
    }

    part def VehicleStateUpdater {
        attribute currentSpeed : Real;
        attribute currentPosition : Real;
        attribute acceleration : Real;
        attribute timeStep : Real;
        attribute newSpeed : Real;
        attribute newPosition : Real;
    }

    port def PowerPort {
        attribute power : Real;
    }

    port def ForcePort {
        attribute force : Real;
    }

    port def StatePort {
        attribute speed : Real;
        attribute position : Real;
        attribute acceleration : Real;
    }

    part def VehicleDynamicsSystem {
        attribute totalMass : Real;
        attribute aeroDragCoefficient : Real;
        attribute rollingResistanceCoefficient : Real;
        attribute frontalArea : Real;
        attribute airDensity : Real = 1.225;
        attribute gravitationalAcceleration : Real = 9.81;
        attribute wheelEndOutputPower : Real;
        attribute initialSpeed : Real;
        attribute initialPosition : Real;
        attribute timeStep : Real;
        attribute currentSpeed : Real;
        attribute currentPosition : Real;
        attribute aeroDragForce : Real;
        attribute rollingResistanceForce : Real;
        attribute powerLostToDrag : Real;
        attribute powerLostToRolling : Real;
        attribute availableDrivingPower : Real;
        attribute netForce : Real;
        attribute acceleration : Real;
        attribute newSpeed : Real;
        attribute newPosition : Real;
        attribute drivingMode : DrivingMode;

        part vehicleParams : VehicleParameters;
        part initialConds : InitialConditions;
        part powerInput : PowerInput;
        part dragCalc : DragForceCalculator;
        part rollingCalc : RollingResistanceCalculator;
        part powerCalc : AvailablePowerCalculator;
        part accelCalc : AccelerationCalculator;
        part stateUpdater : VehicleStateUpdater;

        port powerIn : PowerPort;
        port stateOut : StatePort;
    }

    part vehicleDynamicsInstance : VehicleDynamicsSystem {
        attribute totalMass = 1500.0;
        attribute aeroDragCoefficient = 0.3;
        attribute rollingResistanceCoefficient = 0.015;
        attribute frontalArea = 2.2;
        attribute airDensity = 1.225;
        attribute gravitationalAcceleration = 9.81;
        attribute wheelEndOutputPower = 50000.0;
        attribute initialSpeed = 0.0;
        attribute initialPosition = 0.0;
        attribute timeStep = 0.1;
        attribute currentSpeed = 0.0;
        attribute currentPosition = 0.0;
        attribute drivingMode = DrivingMode::STRAIGHT_LINE;
    }
}