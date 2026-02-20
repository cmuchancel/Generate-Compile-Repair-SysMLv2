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

part def FourCylinderCrankshaft {
    attribute crankshaftID : String = "CRANK-4CYL";
    attribute length : Real = 0.45;
    attribute mass : Real = 12.5;
    attribute material : String = "ForgedSteel";
    attribute numberOfJournals : Integer = 5;
}

part def FourCylinderAssembly1 {
    attribute assemblyID : String = "CYL-ASM-1";
    part cylinder : Cylinder {
        attribute cylinderID : String = "CYL-1";
        attribute bore : Real = 86.0;
        attribute stroke : Real = 86.0;
        attribute displacement : Real = 499.7;
        attribute compressionRatio : Real = 10.5;
    }
    part piston : Piston {
        attribute pistonID : String = "PST-1";
        attribute mass : Real = 0.32;
        attribute diameter : Real = 85.8;
        attribute material : String = "AluminumAlloy";
    }
    part connectingRod : ConnectingRod {
        attribute rodID : String = "ROD-1";
        attribute length : Real = 0.145;
        attribute mass : Real = 0.52;
        attribute material : String = "ForgedSteel";
    }
}

part def FourCylinderAssembly2 {
    attribute assemblyID : String = "CYL-ASM-2";
    part cylinder : Cylinder {
        attribute cylinderID : String = "CYL-2";
        attribute bore : Real = 86.0;
        attribute stroke : Real = 86.0;
        attribute displacement : Real = 499.7;
        attribute compressionRatio : Real = 10.5;
    }
    part piston : Piston {
        attribute pistonID : String = "PST-2";
        attribute mass : Real = 0.32;
        attribute diameter : Real = 85.8;
        attribute material : String = "AluminumAlloy";
    }
    part connectingRod : ConnectingRod {
        attribute rodID : String = "ROD-2";
        attribute length : Real = 0.145;
        attribute mass : Real = 0.52;
        attribute material : String = "ForgedSteel";
    }
}

part def FourCylinderAssembly3 {
    attribute assemblyID : String = "CYL-ASM-3";
    part cylinder : Cylinder {
        attribute cylinderID : String = "CYL-3";
        attribute bore : Real = 86.0;
        attribute stroke : Real = 86.0;
        attribute displacement : Real = 499.7;
        attribute compressionRatio : Real = 10.5;
    }
    part piston : Piston {
        attribute pistonID : String = "PST-3";
        attribute mass : Real = 0.32;
        attribute diameter : Real = 85.8;
        attribute material : String = "AluminumAlloy";
    }
    part connectingRod : ConnectingRod {
        attribute rodID : String = "ROD-3";
        attribute length : Real = 0.145;
        attribute mass : Real = 0.52;
        attribute material : String = "ForgedSteel";
    }
}

part def FourCylinderAssembly4 {
    attribute assemblyID : String = "CYL-ASM-4";
    part cylinder : Cylinder {
        attribute cylinderID : String = "CYL-4";
        attribute bore : Real = 86.0;
        attribute stroke : Real = 86.0;
        attribute displacement : Real = 499.7;
        attribute compressionRatio : Real = 10.5;
    }
    part piston : Piston {
        attribute pistonID : String = "PST-4";
        attribute mass : Real = 0.32;
        attribute diameter : Real = 85.8;
        attribute material : String = "AluminumAlloy";
    }
    part connectingRod : ConnectingRod {
        attribute rodID : String = "ROD-4";
        attribute length : Real = 0.145;
        attribute mass : Real = 0.52;
        attribute material : String = "ForgedSteel";
    }
}

part def FourCylinderEngine {
    attribute engineID : String = "ENG-4CYL-001";
    attribute engineName : String = "FourCylinderEngine";
    attribute engineType : EngineType = EngineType::FourCylinder;
    attribute numberOfCylinders : Integer = 4;

    part crankshaft : FourCylinderCrankshaft;
    part cylinderAssembly1 : FourCylinderAssembly1;
    part cylinderAssembly2 : FourCylinderAssembly2;
    part cylinderAssembly3 : FourCylinderAssembly3;
    part cylinderAssembly4 : FourCylinderAssembly4;
}

part def SixCylinderCrankshaft {
    attribute crankshaftID : String = "CRANK-6CYL";
    attribute length : Real = 0.60;
    attribute mass : Real = 18.0;
    attribute material : String = "ForgedSteel";
    attribute numberOfJournals : Integer = 7;
}

part def SixCylinderAssembly1 {
    attribute assemblyID : String = "CYL6-ASM-1";
    part cylinder : Cylinder {
        attribute cylinderID : String = "CYL6-1";
        attribute bore : Real = 89.0;
        attribute stroke : Real = 90.0;
        attribute displacement : Real = 559.2;