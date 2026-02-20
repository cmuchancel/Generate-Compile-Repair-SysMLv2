package VehiclePowertrainConfiguration {
    private import ScalarValues::*;

    enum def EngineType {
        enum FourCylinder;
        enum SixCylinder;
        enum EightCylinder;
        enum Hybrid;
        enum Electric;
    }

    enum def EvaluationRating {
        enum Poor;
        enum Fair;
        enum Good;
        enum Excellent;
    }

    enum def RecommendationStatus {
        enum NotEvaluated;
        enum Recommended;
        enum NotRecommended;
    }

    part def Cylinder {
        attribute cylinderId : Integer;
        attribute bore : Real;
        attribute stroke : Real;
        attribute displacement : Real;
        attribute compressionRatio : Real;
    }

    part def Piston {
        attribute pistonId : Integer;
        attribute diameter : Real;
        attribute mass : Real;
        attribute material : String;
    }

    part def ConnectingRod {
        attribute rodId : Integer;
        attribute length : Real;
        attribute mass : Real;
        attribute material : String;
    }

    part def Crankshaft {
        attribute crankshaftId : Integer;
        attribute length : Real;
        attribute mass : Real;
        attribute material : String;
        attribute numberOfThrows : Integer;
    }

    part def CylinderAssembly {
        attribute assemblyId : Integer;
        part cylinder : Cylinder;
        part piston : Piston;
        part connectingRod : ConnectingRod;
    }

    part def Engine {
        attribute engineId : Integer;
        attribute engineType : EngineType;
        attribute numberOfCylinders : Integer;
        attribute totalDisplacement : Real;
        attribute maxPower : Real;
        attribute maxTorque : Real;
        attribute totalMass : Real;
        attribute thermalEfficiency : Real;
        attribute fuelConsumption : Real;
        attribute reliabilityScore : Real;
        attribute manufacturingCost : Real;
        attribute maintenanceCost : Real;
        part crankshaft : Crankshaft;
        part cylinderAssemblies : CylinderAssembly[1..8];
    }

    part def FourCylinderEngine {
        attribute engineId : Integer;
        attribute engineType : EngineType = EngineType::FourCylinder;
        attribute numberOfCylinders : Integer = 4;
        attribute totalDisplacement : Real;
        attribute maxPower : Real;
        attribute maxTorque : Real;
        attribute totalMass : Real;
        attribute thermalEfficiency : Real;
        attribute fuelConsumption : Real;
        attribute reliabilityScore : Real;
        attribute manufacturingCost : Real;
        attribute maintenanceCost : Real;
        part crankshaft : Crankshaft;
        part cylinderAssemblies : CylinderAssembly[4];
    }

    part def SixCylinderEngine {
        attribute engineId : Integer;
        attribute engineType : EngineType = EngineType::SixCylinder;
        attribute numberOfCylinders : Integer = 6;
        attribute totalDisplacement : Real;
        attribute maxPower : Real;
        attribute maxTorque : Real;
        attribute totalMass : Real;
        attribute thermalEfficiency : Real;
        attribute fuelConsumption : Real;
        attribute reliabilityScore : Real;
        attribute manufacturingCost : Real;
        attribute maintenanceCost : Real;
        part crankshaft : Crankshaft;
        part cylinderAssemblies : CylinderAssembly[6];
    }

    part def EightCylinderEngine {
        attribute engineId : Integer;
        attribute engineType : EngineType = EngineType::EightCylinder;
        attribute numberOfCylinders : Integer = 8;
        attribute totalDisplacement : Real;
        attribute maxPower : Real;
        attribute maxTorque : Real;
        attribute totalMass : Real;
        attribute thermalEfficiency : Real;
        attribute fuelConsumption : Real;
        attribute reliabilityScore : Real;
        attribute manufacturingCost : Real;
        attribute maintenanceCost : Real;
        part crankshaft : Crankshaft;
        part cylinderAssemblies : CylinderAssembly[8];
    }

    port def PowerOutputPort {
        attribute powerValue : Real;
        attribute torqueValue : Real;
        attribute rotationalSpeed : Real;
    }

    port def FuelInputPort {
        attribute fuelFlowRate : Real;
        attribute fuelPressure : Real;
    }

    port def DataPort {
        attribute dataLabel : String;
        attribute dataValue : Real;
    }

    part def PerformanceAttribute {
        attribute attributeName : String;
        attribute weight : Real;
        attribute rawValue : Real;
        attribute normalizedValue : Real;
        attribute weightedScore : Real;
        attribute rating : EvaluationRating;
    }

    part def TradeoffEvaluationModel {
        attribute modelId : Integer;
        attribute modelName : String;
        attribute powerWeight : Real;
        attribute massWeight : Real;
        attribute efficiencyWeight : Real;
        attribute reliabilityWeight : Real;
        attribute costWeight : Real;
        part powerAttribute : PerformanceAttribute;
        part massAttribute : PerformanceAttribute;
        part efficiencyAttribute : PerformanceAttribute;
        part reliabilityAttribute : PerformanceAttribute;
        part costAttribute : PerformanceAttribute;
    }

    part def EngineEvaluationResult {
        attribute resultId : Integer;
        attribute engineId : Integer;
        attribute engineType : EngineType;
        attribute powerScore : Real;
        attribute massScore : Real;
        attribute efficiencyScore : Real;
        attribute reliabilityScore : Real;
        attribute costScore : Real;
        attribute totalWeightedScore : Real;
        attribute overallRating : EvaluationRating;
        attribute recommendationStatus : RecommendationStatus;
        attribute rankingPosition : Integer;
    }

    part def SelectionExplanation {
        attribute explanationId : Integer;
        attribute selectedEngineType : EngineType;
        attribute selectionRationale : String;
        attribute powerJustification : String;
        attribute massJustification : String;
        attribute efficiencyJustification : String;
        attribute reliabilityJustification : String;
        attribute costJustification : String;
        attribute tradeoffSummary : String;
        attribute confidenceLevel : Real;
    }

    part def OptimalEngineRecommendation {
        attribute recommendationId : Integer;
        attribute recommendedEngineType : EngineType;
        attribute overallScore : Real;
        attribute powerScore : Real;
        attribute massScore : Real;
        attribute efficiencyScore : Real;
        attribute reliabilityScore : Real;
        attribute costScore : Real;
        attribute status : RecommendationStatus;
        part explanation : SelectionExplanation;
    }

    part def PowertrainAnalysisSystem {
        attribute systemId : Integer;
        attribute systemName : String;
        attribute analysisVersion : String;

        part fourCylinderCandidate : FourCylinderEngine;
        part sixCylinderCandidate : SixCylinderEngine;
        part eightCylinderCandidate : EightCylinderEngine;

        part evaluationModel : TradeoffEvaluationModel;

        part fourCylinderResult : EngineEvaluationResult;
        part sixCylinderResult : EngineEvaluationResult;
        part eightCylinderResult : EngineEvaluationResult;

        part optimalRecommendation : OptimalEngineRecommendation;

        port fourCylinderPowerOut : PowerOutputPort;
        port sixCylinderPowerOut : PowerOutputPort;
        port eightCylinderPowerOut : PowerOutputPort;

        port systemFuelIn : FuelInputPort;
        port analysisDataOut : DataPort;
    }

    item def EngineSpecificationData {
        attribute specId : Integer;
        attribute engineType : EngineType;
        attribute displacementSpec : Real;
        attribute powerSpec : Real;
        attribute torqueSpec : Real;
        attribute massSpec : Real;
        attribute efficiencySpec : Real;
        attribute reliabilitySpec : Real;
        attribute costSpec : Real;
    }

    item def EvaluationReport {
        attribute reportId : Integer;
        attribute reportTitle : String;
        attribute analysisDate : String;
        attribute recommendedEngine : EngineType;
        attribute finalScore : Real;
        attribute reportSummary : String;
    }

    part vehiclePowertrainSystem : PowertrainAnalysisSystem {
        attribute systemId = 1;
        attribute systemName = "Vehicle Powertrain Configuration Optimizer";
        attribute analysisVersion = "1.0";

        part fourCylinderCandidate : FourCylinderEngine {
            attribute engineId = 1;
            attribute totalDisplacement = 2.0;
            attribute maxPower = 115.0;
            attribute maxTorque = 200.0;
            attribute totalMass = 140.0;
            attribute thermalEfficiency = 0.38;
            attribute fuelConsumption = 7.5;
            attribute reliabilityScore = 0.88;
            attribute manufacturingCost = 3500.0;
            attribute maintenanceCost = 800.0;

            part crankshaft : Crankshaft {
                attribute crankshaftId = 1;
                attribute length = 0.35;
                attribute mass = 12.0;
                attribute material = "Forged Steel";
                attribute numberOfThrows = 4;
            }

            part cylinderAssemblies : CylinderAssembly[4] {
                attribute assemblyId = 1;
                part cylinder : Cylinder {
                    attribute cylinderId = 1;
                    attribute bore = 86.0;
                    attribute stroke = 86.0;
                    attribute displacement = 0.5;
                    attribute compressionRatio = 10.5;
                }
                part piston : Piston {
                    attribute pistonId = 1;
                    attribute diameter = 86.0;
                    attribute mass = 0.35;
                    attribute material = "Aluminum Alloy";
                }
                part connectingRod : ConnectingRod {
                    attribute rodId = 1;
                    attribute length = 145.0;
                    attribute mass = 0.55;
                    attribute material = "Forged Steel";
                }
            }
        }

        part sixCylinderCandidate : SixCylinderEngine {
            attribute engineId = 2;
            attribute totalDisplacement = 3.0;
            attribute maxPower = 180.0;
            attribute maxTorque = 320.0;
            attribute totalMass = 200.0;
            attribute thermalEfficiency = 0.36;
            attribute fuelConsumption = 10.5;
            attribute reliabilityScore = 0.85;
            attribute manufacturingCost = 5500.0;
            attribute maintenanceCost = 1100.0;

            part crankshaft : Crankshaft {
                attribute crankshaftId = 2;
                attribute length = 0.50;
                attribute mass = 18.0;
                attribute material = "Forged Steel";
                attribute numberOfThrows = 6;
            }

            part cylinderAssemblies : CylinderAssembly[6] {
                attribute assemblyId = 2;
                part cylinder : Cylinder {
                    attribute cylinderId = 2;
                    attribute bore = 89.0;
                    attribute stroke = 80.0;
                    attribute displacement = 0.5;
                    attribute compr