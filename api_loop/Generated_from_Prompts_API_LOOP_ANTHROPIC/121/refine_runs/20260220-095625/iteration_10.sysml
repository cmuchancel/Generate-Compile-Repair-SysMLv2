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
    attribute cylinderID : String;
    attribute bore : Real;
    attribute stroke : Real;
    attribute displacement : Real;
    attribute compressionRatio : Real;
}

part def Piston {
    attribute pistonID : String;
    attribute mass : Real;
    attribute diameter : Real;
    attribute material : String;
}

part def ConnectingRod {
    attribute rodID : String;
    attribute length : Real;
    attribute mass : Real;
    attribute material : String;
}

part def Crankshaft {
    attribute crankshaftID : String;
    attribute length : Real;
    attribute mass : Real;
    attribute material : String;
    attribute numberOfJournals : Integer;
}

part def CylinderAssembly {
    attribute assemblyID : String;
    part cylinder : Cylinder;
    part piston : Piston;
    part connectingRod : ConnectingRod;
}

part def EngineAssembly {
    attribute engineID : String;
    attribute engineName : String;
    attribute engineType : EngineType;
    attribute numberOfCylinders : Integer;
    part crankshaft : Crankshaft;
    part cylinderAssemblies : CylinderAssembly[1..*];
}

part def PerformanceData {
    attribute peakPower : Real;
    attribute peakTorque : Real;
    attribute redlineRPM : Real;
    attribute idleRPM : Real;
    attribute fuelConsumption : Real;
    attribute thermalEfficiency : Real;
    attribute emissionsCO2 : Real;
}

part def MassData {
    attribute engineMass : Real;
    attribute dryWeight : Real;
    attribute wetWeight : Real;
}

part def ReliabilityData {
    attribute meanTimeBetweenFailures : Real;
    attribute expectedLifespan : Real;
    attribute maintenanceIntervalKm : Real;
    attribute warrantyYears : Real;
}

part def CostData {
    attribute manufacturingCost : Real;
    attribute maintenanceCostPerYear : Real;
    attribute fuelCostPerKm : Real;
    attribute totalCostOfOwnership : Real;
}

part def EngineScheme {
    attribute schemeID : String;
    attribute schemeName : String;
    attribute engineType : EngineType;
    part engineAssembly : EngineAssembly;
    part performanceData : PerformanceData;
    part massData : MassData;
    part reliabilityData : ReliabilityData;
    part costData : CostData;
}

part def TradeoffWeights {
    attribute powerWeight : Real;
    attribute massWeight : Real;
    attribute efficiencyWeight : Real;
    attribute reliabilityWeight : Real;
    attribute costWeight : Real;
}

part def NormalizedScores {
    attribute powerScore : Real;
    attribute massScore : Real;
    attribute efficiencyScore : Real;
    attribute reliabilityScore : Real;
    attribute costScore : Real;
    attribute compositeScore : Real;
}

part def EvaluationResult {
    attribute resultID : String;
    attribute schemeID : String;
    attribute schemeName : String;
    attribute engineType : EngineType;
    attribute powerRating : EvaluationRating;
    attribute massRating : EvaluationRating;
    attribute efficiencyRating : EvaluationRating;
    attribute reliabilityRating : EvaluationRating;
    attribute costRating : EvaluationRating;
    attribute overallRating : EvaluationRating;
    part scores : NormalizedScores;
    attribute recommendationStatus : RecommendationStatus;
    attribute rationaleExplanation : String;
}

part def EvaluationModel {
    attribute modelID : String;
    attribute modelName : String;
    attribute modelVersion : String;
    part weights : TradeoffWeights;
}

part def OptimalRecommendation {
    attribute recommendationID : String;
    attribute recommendedSchemeID : String;
    attribute recommendedSchemeName : String;
    attribute recommendedEngineType : EngineType;
    attribute compositeScore : Real;
    attribute selectionRationale : String;
    attribute decisionTransparencyNote : String;
    attribute traceabilityReference : String;
}

port def DataInputPort {
    attribute dataSource : String;
}

port def EvaluationOutputPort {
    attribute outputTarget : String;
}

port def RecommendationOutputPort {
    attribute outputTarget : String;
}

part def PowertrainConfigurationSystem {
    attribute systemID : String;
    attribute systemName : String;
    attribute systemVersion : String;

    port dataInput : DataInputPort;
    port evaluationOutput : EvaluationOutputPort;
    port recommendationOutput : RecommendationOutputPort;

    part evaluationModel : EvaluationModel;
    part optimalRecommendation : OptimalRecommendation;

    part fourCylinderScheme : EngineScheme;
    part sixCylinderScheme : EngineScheme;
    part eightCylinderScheme : EngineScheme;
    part hybridScheme : EngineScheme;
    part electricScheme : EngineScheme;

    part fourCylinderResult : EvaluationResult;
    part sixCylinderResult : EvaluationResult;
    part eightCylinderResult : EvaluationResult;
    part hybridResult : EvaluationResult;
    part electricResult : EvaluationResult;
}

part system1 : PowertrainConfigurationSystem;

part evaluationModelInst : EvaluationModel {
    attribute modelID = "EM-001";
    attribute modelName = "MultiAttributeTradeoffModel";
    attribute modelVersion = "1.0";
    part weights : TradeoffWeights {
        attribute powerWeight = 0.25;
        attribute massWeight = 0.15;
        attribute efficiencyWeight = 0.25;
        attribute reliabilityWeight = 0.20;
        attribute costWeight = 0.15;
    }
}

part optimalRecommendationInst : OptimalRecommendation {
    attribute recommendationID = "REC-001";
    attribute recommendedSchemeID = "SCH-HYB-001";
    attribute recommendedSchemeName = "HybridEngineScheme";
    attribute recommendedEngineType = EngineType::Hybrid;
    attribute compositeScore = 0.87;
    attribute selectionRationale = "Hybrid engine achieves best balance of efficiency and cost";
    attribute decisionTransparencyNote = "Weighted scoring across power mass efficiency reliability cost";
    attribute traceabilityReference = "EvaluationModel EM-001 version 1.0";
}

part fourCylinderSchemeInst : EngineScheme {
    attribute schemeID = "SCH-4CYL-001";
    attribute schemeName = "FourCylinderScheme";
    attribute engineType = EngineType::FourCylinder;
    part engineAssembly : EngineAssembly {
        attribute engineID = "ENG-4CYL-001";
        attribute engineName = "FourCylinderEngine";
        attribute engineType = EngineType::FourCylinder;
        attribute numberOfCylinders = 4;
        part crankshaft : Crankshaft {
            attribute crankshaftID = "CRANK-4CYL";
            attribute length = 0.45;
            attribute mass = 12.5;
            attribute material = "ForgedSteel";
            attribute numberOfJournals = 5;
        }
        part cylinderAssemblies : CylinderAssembly {
            attribute assemblyID = "CYL-ASM-4CYL";
            part cylinder : Cylinder {
                attribute cylinderID = "CYL-4CYL-1";
                attribute bore = 86.0;
                attribute stroke = 86.0;
                attribute displacement = 499.7;
                attribute compressionRatio = 10.5;
            }
            part piston : Piston {
                attribute pistonID = "PST-4CYL-1";
                attribute mass = 0.32;
                attribute diameter = 85.8;
                attribute material = "AluminumAlloy";
            }
            part connectingRod : ConnectingRod {
                attribute rodID = "ROD-4CYL-1";
                attribute length = 0.145;
                attribute mass = 0.52;
                attribute material = "ForgedSteel";
            }
        }
    }
    part performanceData : PerformanceData {
        attribute peakPower = 110.0;
        attribute peakTorque = 200.0;
        attribute redlineRPM = 6500.0;
        attribute idleRPM = 800.0;
        attribute fuelConsumption = 7.2;
        attribute thermalEfficiency = 0.36;
        attribute emissionsCO2 = 165.0;
    }
    part massData : MassData {
        attribute engineMass = 135.0;
        attribute dryWeight = 130.0;
        attribute wetWeight = 145.0;
    }
    part reliabilityData : ReliabilityData {
        attribute meanTimeBetweenFailures = 8000.0;
        attribute expectedLifespan = 250000.0;
        attribute maintenanceIntervalKm = 15000.0;
        attribute warrantyYears = 3.0;
    }
    part costData : CostData {
        attribute manufacturingCost = 3500.0;
        attribute maintenanceCostPerYear = 400.0;
        attribute fuelCostPerKm = 0.09;
        attribute totalCostOfOwnership = 12000.0;
    }
}

part sixCylinderSchemeInst : EngineScheme {
    attribute schemeID = "SCH-6CYL-001";
    attribute schemeName = "SixCylinderScheme";
    attribute engineType = EngineType::SixCylinder;
    part engineAssembly : EngineAssembly {
        attribute engineID = "ENG-6CYL-001";
        attribute engineName = "SixCylinderEngine";
        attribute engineType = EngineType::SixCylinder;
        attribute numberOfCylinders = 6;
        part crankshaft : Crankshaft {
            attribute crankshaftID = "CRANK-6CYL";
            attribute length = 0.60;
            attribute mass = 18.0;
            attribute material = "ForgedSteel";
            attribute numberOfJournals = 7;
        }
        part cylinderAssemblies : CylinderAssembly {
            attribute assemblyID = "CYL-ASM-6CYL";
            part cylinder : Cylinder {
                attribute cylinderID = "CYL-6CYL-1";
                attribute bore = 89.0;
                attribute stroke = 90.0;
                attribute displacement = 559.6;
                attribute compressionRatio = 11.0;
            }
            part piston : Piston {
                attribute pistonID = "PST-6CYL-1";