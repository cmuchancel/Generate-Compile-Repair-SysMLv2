package VehicleDynamicsSimulation {
    private import ScalarValues::*;

    enum def SimulationStatus {
        enum Ready;
        enum Running;
        enum Complete;
        enum Error;
    }

    item def VehicleInputData {
        attribute initialVelocity : Real;
        attribute initialPosition : Real;
        attribute acceleration : Real;
        attribute timeStep : Real;
    }

    item def VehicleOutputData {
        attribute finalVelocity : Real;
        attribute finalPosition : Real;
    }

    port def InputPort {
        attribute initialVelocity : Real;
        attribute initialPosition : Real;
        attribute acceleration : Real;
        attribute timeStep : Real;
    }

    port def OutputPort {
        attribute finalVelocity : Real;
        attribute finalPosition : Real;
    }

    port def AnalysisToolPort {
        attribute toolName : String;
        attribute connected : Boolean;
    }

    part def Vehicle {
        attribute mass : Real;
        attribute currentVelocity : Real;
        attribute currentPosition : Real;
        attribute currentAcceleration : Real;
    }

    part def ModelCenterAnalysisTool {
        attribute toolName : String = "ModelCenter";
        attribute version : String;
        attribute isConnected : Boolean;

        port analysisInput : InputPort;
        port analysisOutput : OutputPort;
    }

    part def SimulationEngine {
        attribute simulationName : String;
        attribute status : SimulationStatus;
        attribute initialVelocity : Real;
        attribute initialPosition : Real;
        attribute acceleration : Real;
        attribute timeStep : Real;
        attribute computedFinalVelocity : Real;
        attribute computedFinalPosition : Real;

        port simInput : InputPort;
        port simOutput : OutputPort;
        port toolConnection : AnalysisToolPort;
    }

    part def DataTransferManager {
        attribute transferReliability : Real;
        attribute dataValidated : Boolean;
        attribute inputVelocity : Real;
        attribute inputPosition : Real;
        attribute inputAcceleration : Real;
        attribute inputTimeStep : Real;
        attribute outputVelocity : Real;
        attribute outputPosition : Real;

        port dataIn : InputPort;
        port dataOut : OutputPort;
    }

    part def VehicleDynamicsSystem {
        attribute systemName : String = "VehicleDynamicsSimulationSystem";
        attribute description : String = "Automated simulation and analysis of vehicle dynamic behavior";

        part vehicle : Vehicle {
            attribute mass : Real = 1500.0;
            attribute currentVelocity : Real = 0.0;
            attribute currentPosition : Real = 0.0;
            attribute currentAcceleration : Real = 0.0;
        }

        part simulationEngine : SimulationEngine {
            attribute simulationName : String = "VehicleDynamicsSim";
            attribute status : SimulationStatus = SimulationStatus::Ready;
            attribute initialVelocity : Real = 0.0;
            attribute initialPosition : Real = 0.0;
            attribute acceleration : Real = 0.0;
            attribute timeStep : Real = 0.01;
            attribute computedFinalVelocity : Real = 0.0;
            attribute computedFinalPosition : Real = 0.0;
        }

        part analysisToolIntegration : ModelCenterAnalysisTool {
            attribute toolName : String = "ModelCenter";
            attribute version : String = "2024";
            attribute isConnected : Boolean = true;
        }

        part dataTransferManager : DataTransferManager {
            attribute transferReliability : Real = 1.0;
            attribute dataValidated : Boolean = false;
            attribute inputVelocity : Real = 0.0;
            attribute inputPosition : Real = 0.0;
            attribute inputAcceleration : Real = 0.0;
            attribute inputTimeStep : Real = 0.01;
            attribute outputVelocity : Real = 0.0;
            attribute outputPosition : Real = 0.0;
        }

        connect simulationEngine.simOutput to dataTransferManager.dataIn;
        connect dataTransferManager.dataOut to analysisToolIntegration.analysisInput;
        connect analysisToolIntegration.analysisOutput to simulationEngine.simInput;
    }
}