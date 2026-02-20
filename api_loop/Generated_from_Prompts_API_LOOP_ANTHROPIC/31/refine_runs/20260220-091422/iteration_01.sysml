package VehicleSimulation {
    private import ScalarValues::*;

    enum def SimulationStatus {
        enum Running;
        enum Completed;
        enum Error;
    }

    item def PowerInput {
        attribute timeStep : Real;
        attribute powerValue : Real;
    }

    item def SimulationStep {
        attribute stepIndex : Real;
        attribute time : Real;
        attribute velocity : Real;
        attribute position : Real;
        attribute appliedPower : Real;
    }

    part def VehicleParameters {
        attribute mass : Real;
        attribute initialPosition : Real;
        attribute initialVelocity : Real;
        attribute timeInterval : Real;
    }

    part def PowerSequence {
        attribute totalSteps : Real;
        part powerInputs : PowerInput;
    }

    part def MotionCalculator {
        attribute currentTime : Real;
        attribute currentVelocity : Real;
        attribute currentPosition : Real;
        attribute currentForce : Real;
        attribute currentAcceleration : Real;
        attribute deltaVelocity : Real;
        attribute deltaPosition : Real;
    }

    part def TrajectoryOutput {
        attribute totalSteps : Real;
        attribute finalPosition : Real;
        attribute finalVelocity : Real;
        attribute finalTime : Real;
        part steps : SimulationStep;
    }

    port def SimulationInputPort {
        attribute mass : Real;
        attribute initialPosition : Real;
        attribute initialVelocity : Real;
        attribute timeInterval : Real;
        attribute totalSteps : Real;
    }

    port def SimulationOutputPort {
        attribute finalPosition : Real;
        attribute finalVelocity : Real;
        attribute finalTime : Real;
        attribute status : Real;
    }

    port def PowerDataPort {
        attribute stepIndex : Real;
        attribute powerValue : Real;
    }

    part def VehicleSimulator {
        attribute simulatorId : String;
        attribute status : SimulationStatus;

        port inputPort : SimulationInputPort;
        port outputPort : SimulationOutputPort;
        port powerPort : PowerDataPort;

        part vehicleParams : VehicleParameters;
        part powerSequence : PowerSequence;
        part motionCalc : MotionCalculator;
        part trajectory : TrajectoryOutput;
    }

    part def SimulationSystem {
        attribute systemName : String;
        attribute systemVersion : String;

        part simulator : VehicleSimulator;
        part parameters : VehicleParameters;
        part power : PowerSequence;
        part output : TrajectoryOutput;
    }

    part simulationSystem : SimulationSystem {
        attribute systemName = "VehicleMotionSimulator";
        attribute systemVersion = "1.0";

        part simulator : VehicleSimulator {
            attribute simulatorId = "SIM-001";
            attribute status = SimulationStatus::Running;

            part vehicleParams : VehicleParameters {
                attribute mass = 1000.0;
                attribute initialPosition = 0.0;
                attribute initialVelocity = 0.0;
                attribute timeInterval = 0.1;
            }

            part powerSequence : PowerSequence {
                attribute totalSteps = 10.0;
            }

            part motionCalc : MotionCalculator {
                attribute currentTime = 0.0;
                attribute currentVelocity = 0.0;
                attribute currentPosition = 0.0;
                attribute currentForce = 0.0;
                attribute currentAcceleration = 0.0;
                attribute deltaVelocity = 0.0;
                attribute deltaPosition = 0.0;
            }

            part trajectory : TrajectoryOutput {
                attribute totalSteps = 10.0;
                attribute finalPosition = 0.0;
                attribute finalVelocity = 0.0;
                attribute finalTime = 1.0;
            }
        }
    }
}