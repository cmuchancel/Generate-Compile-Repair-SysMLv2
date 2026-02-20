package VehicleDynamicsSimulation {
    private import ScalarValues::*;

    enum def SimulationStatus {
        enum INITIALIZING;
        enum RUNNING;
        enum COMPLETED;
        enum ERROR;
    }

    enum def DriveMode {
        enum ACCELERATION;
        enum CRUISE;
        enum DECELERATION;
        enum BRAKING;
    }

    item def VehicleParameters {
        attribute vehicleMass : Real;
        attribute wheelPower : Real;
        attribute aeroDragCoefficient : Real;
        attribute rollingResistanceCoefficient : Real;
    }

    item def InitialConditions {
        attribute initialSpeed : Real;
        attribute initialPosition : Real;
    }

    item def DynamicState {
        attribute currentSpeed : Real;
        attribute currentPosition : Real;
        attribute currentAcceleration : Real;
        attribute currentDisplacement : Real;
        attribute timeStep : Real;
        attribute elapsedTime : Real;
    }

    item def ForceComponents {
        attribute driveForce : Real;
        attribute aeroDragForce : Real;
        attribute rollingResistanceForce : Real;
        attribute netForce : Real;
    }

    item def SimulationResult {
        attribute velocityProfile : Real;
        attribute accelerationProfile : Real;
        attribute displacementProfile : Real;
        attribute positionProfile : Real;
        attribute simulationDuration : Real;
    }

    port def InputPort {
        attribute dataValue : Real;
        attribute dataLabel : String;
    }

    port def OutputPort {
        attribute resultValue : Real;
        attribute resultLabel : String;
    }

    port def ControlPort {
        attribute commandValue : Real;
        attribute commandLabel : String;
    }

    part def UserInputHandler {
        attribute massInput : Real;
        attribute powerInput : Real;
        attribute dragCoefficientInput : Real;
        attribute rollingCoefficientInput : Real;
        attribute initialSpeedInput : Real;
        attribute initialPositionInput : Real;

        port inputReceiver : InputPort;
        port outputSender : OutputPort;
    }

    part def ParameterValidator {
        attribute validatedMass : Real;
        attribute validatedPower : Real;
        attribute validatedDragCoefficient : Real;
        attribute validatedRollingCoefficient : Real;
        attribute validationStatus : Boolean;
        attribute massLowerBound : Real = 100.0;
        attribute massUpperBound : Real = 100000.0;
        attribute powerLowerBound : Real = 0.0;
        attribute dragCoefficientLowerBound : Real = 0.0;
        attribute rollingCoefficientLowerBound : Real = 0.0;

        port validationInput : InputPort;
        port validationOutput : OutputPort;
    }

    part def ForceCalculator {
        attribute airDensity : Real = 1.225;
        attribute gravity : Real = 9.81;
        attribute frontalArea : Real;
        attribute computedDriveForce : Real;
        attribute computedAeroDragForce : Real;
        attribute computedRollingResistanceForce : Real;
        attribute computedNetForce : Real;

        port forceInput : InputPort;
        port forceOutput : OutputPort;
    }

    part def AccelerationComputer {
        attribute computedAcceleration : Real;
        attribute netForceIn : Real;
        attribute massIn : Real;

        port accelInput : InputPort;
        port accelOutput : OutputPort;
    }

    part def VelocityIntegrator {
        attribute previousVelocity : Real;
        attribute newVelocity : Real;
        attribute deltaTime : Real = 0.01;
        attribute accelerationIn : Real;

        port velocityInput : InputPort;
        port velocityOutput : OutputPort;
    }

    part def PositionIntegrator {
        attribute previousPosition : Real;
        attribute newPosition : Real;
        attribute displacementDelta : Real;
        attribute deltaTime : Real = 0.01;
        attribute velocityIn : Real;

        port positionInput : InputPort;
        port positionOutput : OutputPort;
    }

    part def StateUpdater {
        attribute updatedSpeed : Real;
        attribute updatedPosition : Real;
        attribute updatedAcceleration : Real;
        attribute updatedDisplacement : Real;
        attribute currentTimeStep : Real;
        attribute totalElapsedTime : Real;
        attribute simulationMode : DriveMode;

        port stateInput : InputPort;
        port stateOutput : OutputPort;
    }

    part def SimulationController {
        attribute simulationStatus : SimulationStatus;
        attribute totalSimulationTime : Real = 100.0;
        attribute timeStepSize : Real = 0.01;
        attribute currentStep : Real;
        attribute maxSteps : Real;

        port controlInput : ControlPort;
        port controlOutput : ControlPort;
    }

    part def OutputRenderer {
        attribute renderedVelocity : Real;
        attribute renderedAcceleration : Real;
        attribute renderedDisplacement : Real;
        attribute renderedPosition : Real;
        attribute renderedTime : Real;

        port renderInput : InputPort;
        port renderOutput : OutputPort;
    }

    part def PerformanceAnalyzer {
        attribute peakVelocity : Real;
        attribute peakAcceleration : Real;
        attribute totalDistanceTraveled : Real;
        attribute averageVelocity : Real;
        attribute averageAcceleration : Real;
        attribute energyConsumed : Real;

        port analysisInput : InputPort;
        port analysisOutput : OutputPort;
    }

    part def VehicleDynamicsSystem {
        attribute systemName : String = "VehicleDynamicsSimulator";
        attribute systemVersion : Real = 1.0;
        attribute airDensityConstant : Real = 1.225;
        attribute gravityConstant : Real = 9.81;

        part userInput : UserInputHandler;
        part paramValidator : ParameterValidator;
        part forceCalc : ForceCalculator;
        part accelComputer : AccelerationComputer;
        part velocityInteg : VelocityIntegrator;
        part positionInteg : PositionIntegrator;
        part stateUpdate : StateUpdater;
        part simController : SimulationController;
        part outputRender : OutputRenderer;
        part perfAnalyzer : PerformanceAnalyzer;

        connect userInput.outputSender to paramValidator.validationInput;
        connect paramValidator.validationOutput to forceCalc.forceInput;
        connect forceCalc.forceOutput to accelComputer.accelInput;
        connect accelComputer.accelOutput to velocityInteg.velocityInput;
        connect velocityInteg.velocityOutput to positionInteg.positionInput;
        connect positionInteg.positionOutput to stateUpdate.stateInput;
        connect stateUpdate.stateOutput to outputRender.renderInput;
        connect outputRender.renderOutput to perfAnalyzer.analysisInput;
        connect simController.controlOutput to stateUpdate.stateInput;
    }
}