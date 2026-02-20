package VehicleLongitudinalDynamics {
    private import ScalarValues::*;

    enum def MotionPhase {
        enum Accelerating;
        enum Cruising;
        enum Decelerating;
    }

    part def VehicleMassParameters {
        attribute totalMass : Real;
        attribute rollingResistanceCoeff : Real;
        attribute aeroDragCoeff : Real;
        attribute frontalArea : Real;
        attribute airDensity : Real;
    }

    part def PowerSegment {
        attribute segmentIndex : Integer;
        attribute powerOutput : Real;
        attribute segmentDuration : Real;
        attribute startTime : Real;
        attribute endTime : Real;
    }

    part def InitialConditions {
        attribute initialPosition : Real;
        attribute initialVelocity : Real;
        attribute timeStep : Real;
    }

    part def DynamicState {
        attribute currentTime : Real;
        attribute currentAcceleration : Real;
        attribute currentVelocity : Real;
        attribute currentPosition : Real;
        attribute currentPower : Real;
        attribute tractionForce : Real;
        attribute resistanceForce : Real;
        attribute netForce : Real;
        attribute phase : MotionPhase;
    }

    part def AccelerationSequence {
        attribute stepIndex : Integer;
        attribute timeStamp : Real;
        attribute accelerationValue : Real;
    }

    part def VelocitySequence {
        attribute stepIndex : Integer;
        attribute timeStamp : Real;
        attribute velocityValue : Real;
    }

    part def PositionSequence {
        attribute stepIndex : Integer;
        attribute timeStamp : Real;
        attribute positionValue : Real;
    }

    port def PowerInputPort {
        attribute powerValue : Real;
        attribute timeReference : Real;
    }

    port def DynamicsOutputPort {
        attribute outputAcceleration : Real;
        attribute outputVelocity : Real;
        attribute outputPosition : Real;
        attribute outputTime : Real;
    }

    port def ParameterInputPort {
        attribute mass : Real;
        attribute rollingCoeff : Real;
        attribute dragCoeff : Real;
        attribute area : Real;
        attribute density : Real;
    }

    part def DynamicsCalculationEngine {
        attribute gravity : Real = 9.81;
        attribute simulationTime : Real;
        attribute numberOfSteps : Integer;
        attribute timeStep : Real;

        port powerIn : PowerInputPort;
        port paramIn : ParameterInputPort;
        port dynamicsOut : DynamicsOutputPort;
    }

    part def SimulationController {
        attribute totalSimulationDuration : Real;
        attribute timeStep : Real;
        attribute numberOfSegments : Integer;

        port controlPowerOut : PowerInputPort;
        port controlParamOut : ParameterInputPort;
        port controlDynamicsIn : DynamicsOutputPort;
    }

    part def OutputCollector {
        attribute totalSteps : Integer;

        port dataIn : DynamicsOutputPort;
    }

    part def VehicleLongitudinalSystem {
        attribute systemName : String = "VehicleLongitudinalDynamicsSystem";
        attribute version : String = "1.0";

        part massParams : VehicleMassParameters {
            attribute totalMass : Real = 1500.0;
            attribute rollingResistanceCoeff : Real = 0.015;
            attribute aeroDragCoeff : Real = 0.3;
            attribute frontalArea : Real = 2.2;
            attribute airDensity : Real = 1.225;
        }

        part initConditions : InitialConditions {
            attribute initialPosition : Real = 0.0;
            attribute initialVelocity : Real = 0.0;
            attribute timeStep : Real = 0.1;
        }

        part powerSeg1 : PowerSegment {
            attribute segmentIndex : Integer = 1;
            attribute powerOutput : Real = 50000.0;
            attribute segmentDuration : Real = 10.0;
            attribute startTime : Real = 0.0;
            attribute endTime : Real = 10.0;
        }

        part powerSeg2 : PowerSegment {
            attribute segmentIndex : Integer = 2;
            attribute powerOutput : Real = 30000.0;
            attribute segmentDuration : Real = 10.0;
            attribute startTime : Real = 10.0;
            attribute endTime : Real = 20.0;
        }

        part powerSeg3 : PowerSegment {
            attribute segmentIndex : Integer = 3;
            attribute powerOutput : Real = 10000.0;
            attribute segmentDuration : Real = 10.0;
            attribute startTime : Real = 20.0;
            attribute endTime : Real = 30.0;
        }

        part dynamicsState : DynamicState {
            attribute currentTime : Real = 0.0;
            attribute currentAcceleration : Real = 0.0;
            attribute currentVelocity : Real = 0.0;
            attribute currentPosition : Real = 0.0;
            attribute currentPower : Real = 0.0;
            attribute tractionForce : Real = 0.0;
            attribute resistanceForce : Real = 0.0;
            attribute netForce : Real = 0.0;
            attribute phase : MotionPhase = MotionPhase::Accelerating;
        }

        part calcEngine : DynamicsCalculationEngine {
            attribute gravity : Real = 9.81;
            attribute simulationTime : Real = 30.0;
            attribute numberOfSteps : Integer = 300;
            attribute timeStep : Real = 0.1;
        }

        part simController : SimulationController {
            attribute totalSimulationDuration : Real = 30.0;
            attribute timeStep : Real = 0.1;
            attribute numberOfSegments : Integer = 3;
        }

        part outputCollector : OutputCollector {
            attribute totalSteps : Integer = 300;
        }

        part accelOutput : AccelerationSequence {
            attribute stepIndex : Integer = 0;
            attribute timeStamp : Real = 0.0;
            attribute accelerationValue : Real = 0.0;
        }

        part velOutput : VelocitySequence {
            attribute stepIndex : Integer = 0;
            attribute timeStamp : Real = 0.0;
            attribute velocityValue : Real = 0.0;
        }

        part posOutput : PositionSequence {
            attribute stepIndex : Integer = 0;
            attribute timeStamp : Real = 0.0;
            attribute positionValue : Real = 0.0;
        }

        connect simController.controlPowerOut to calcEngine.powerIn;
        connect simController.controlParamOut to calcEngine.paramIn;
        connect calcEngine.dynamicsOut to simController.controlDynamicsIn;
        connect calcEngine.dynamicsOut to outputCollector.dataIn;
    }
}