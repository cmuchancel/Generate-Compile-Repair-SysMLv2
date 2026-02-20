package VehicleLongitudinalDynamics {
    private import ScalarValues::*;

    enum def MotionState {
        enum Accelerating;
        enum Constant;
        enum Decelerating;
    }

    part def VehicleInputs {
        attribute initialPosition : Real;
        attribute initialVelocity : Real;
        attribute vehicleMass : Real;
        attribute wheelEndPower : Real;
        attribute timeInterval : Real;
    }

    part def EnvironmentParameters {
        attribute dragCoefficient : Real;
        attribute frictionCoefficient : Real;
        attribute airDensity : Real;
        attribute frontalArea : Real;
        attribute gravitationalAcceleration : Real;
    }

    part def DynamicsCalculator {
        attribute driveForce : Real;
        attribute airResistanceForce : Real;
        attribute frictionResistanceForce : Real;
        attribute netForce : Real;
        attribute acceleration : Real;
        attribute finalVelocity : Real;
        attribute finalPosition : Real;
        attribute currentMotionState : MotionState;
    }

    part def VehicleDynamicsSystem {
        part inputs : VehicleInputs;
        part environment : EnvironmentParameters;
        part calculator : DynamicsCalculator;
    }

    port def PowerInputPort {
        attribute power : Real;
        attribute velocity : Real;
    }

    port def ForceOutputPort {
        attribute force : Real;
    }

    port def MotionStatePort {
        attribute position : Real;
        attribute velocity : Real;
        attribute acceleration : Real;
    }

    part def PowerToForceConverter {
        attribute inputPower : Real;
        attribute inputVelocity : Real;
        attribute outputForce : Real;
        port powerIn : PowerInputPort;
        port forceOut : ForceOutputPort;
    }

    part def ResistanceForceCalculator {
        attribute mass : Real;
        attribute velocity : Real;
        attribute dragCoeff : Real;
        attribute frictionCoeff : Real;
        attribute airDensity : Real;
        attribute frontalArea : Real;
        attribute gravity : Real;
        attribute airDrag : Real;
        attribute rollingFriction : Real;
        attribute totalResistance : Real;
        port motionIn : MotionStatePort;
        port forceOut : ForceOutputPort;
    }

    part def AccelerationSolver {
        attribute netForce : Real;
        attribute mass : Real;
        attribute computedAcceleration : Real;
        port forceIn : ForceOutputPort;
        port motionOut : MotionStatePort;
    }

    part def VelocityPositionIntegrator {
        attribute initialVelocity : Real;
        attribute initialPosition : Real;
        attribute acceleration : Real;
        attribute timeStep : Real;
        attribute finalVelocity : Real;
        attribute finalPosition : Real;
        attribute deltaVelocity : Real;
        attribute deltaPosition : Real;
        port stateIn : MotionStatePort;
        port stateOut : MotionStatePort;
    }

    part def SimulationEngine {
        part powerConverter : PowerToForceConverter;
        part resistanceCalc : ResistanceForceCalculator;
        part accelerationSolver : AccelerationSolver;
        part integrator : VelocityPositionIntegrator;

        connect powerConverter.forceOut to accelerationSolver.forceIn;
        connect resistanceCalc.forceOut to accelerationSolver.forceIn;
        connect accelerationSolver.motionOut to integrator.stateIn;
    }

    part def PhysicsConstraints {
        attribute massPositiveCheck : Real;
        attribute timePositiveCheck : Real;
        attribute powerNonNegativeCheck : Real;
        attribute velocityNonNegativeCheck : Real;
        attribute dragCoefficientRangeCheck : Real;
        attribute frictionCoefficientRangeCheck : Real;
    }

    part def MotionEquationSet {
        attribute driveForceEquation : Real;
        attribute airResistanceEquation : Real;
        attribute frictionResistanceEquation : Real;
        attribute newtonSecondLaw : Real;
        attribute kinematicVelocity : Real;
        attribute kinematicPosition : Real;
    }

    part vehicleDynamicsSystem : VehicleDynamicsSystem;
    part simulationEngine : SimulationEngine;
    part physicsConstraints : PhysicsConstraints;
    part motionEquations : MotionEquationSet;
}