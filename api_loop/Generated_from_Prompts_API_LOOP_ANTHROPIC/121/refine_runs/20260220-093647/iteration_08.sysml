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

part fourCylinderCrankshaft : Crankshaft {
    attribute crankshaftID = "CRANK-4CYL";
    attribute length = 0.45;
    attribute mass = 12.5;
    attribute material = "ForgedSteel";
    attribute numberOfJournals = 5;
}

part fourCylinder1 : Cylinder {
    attribute cylinderID = "CYL-1";
    attribute bore = 86.0;
    attribute stroke = 86.0;
    attribute displacement = 499.7;
    attribute compressionRatio = 10.5;
}

part fourPiston1 : Piston {
    attribute pistonID = "PST-1";
    attribute mass = 0.32;
    attribute diameter = 85.8;
    attribute material = "AluminumAlloy";
}

part fourRod1 : ConnectingRod {
    attribute rodID = "ROD-1";
    attribute length = 0.145;
    attribute mass = 0.52;
    attribute material = "ForgedSteel";
}

part fourCylinder2 : Cylinder {
    attribute cylinderID = "CYL-2";
    attribute bore = 86.0;
    attribute stroke = 86.0;
    attribute displacement = 499.7;
    attribute compressionRatio = 10.5;
}

part fourPiston2 : Piston {
    attribute pistonID = "PST-2";
    attribute mass = 0.32;
    attribute diameter = 85.8;
    attribute material = "AluminumAlloy";
}

part fourRod2 : ConnectingRod {
    attribute rodID = "ROD-2";
    attribute length = 0.145;
    attribute mass = 0.52;
    attribute material = "ForgedSteel";
}

part fourCylinder3 : Cylinder {
    attribute cylinderID = "CYL-3";
    attribute bore = 86.0;
    attribute stroke = 86.0;
    attribute displacement = 499.7;
    attribute compressionRatio = 10.5;
}

part fourPiston3 : Piston {
    attribute pistonID = "PST-3";
    attribute mass = 0.32;
    attribute diameter = 85.8;
    attribute material = "AluminumAlloy";
}

part fourRod3 : ConnectingRod {
    attribute rodID = "ROD-3";
    attribute length = 0.145;
    attribute mass = 0.52;
    attribute material = "ForgedSteel";
}

part fourCylinder4 : Cylinder {
    attribute cylinderID = "CYL-4";
    attribute bore = 86.0;
    attribute stroke = 86.0;
    attribute displacement = 499.7;
    attribute compressionRatio = 10.5;
}

part fourPiston4 : Piston {
    attribute pistonID = "PST-4";
    attribute mass = 0.32;
    attribute diameter = 85.8;
    attribute material = "AluminumAlloy";
}

part fourRod4 : ConnectingRod {
    attribute rodID = "ROD-4";
    attribute length = 0.145;
    attribute mass = 0.52;
    attribute material = "ForgedSteel";
}

part fourCylinderAssembly1 : CylinderAssembly {
    attribute assemblyID = "CYL-ASM-1";
    part cylinder : Cylinder;
    part piston : Piston;
    part connectingRod : ConnectingRod;
}

part fourCylinderAssembly2 : CylinderAssembly {
    attribute assemblyID = "CYL-ASM-2";
    part cylinder : Cylinder;
    part piston : Piston;
    part connectingRod : ConnectingRod;
}

part fourCylinderAssembly3 : CylinderAssembly {
    attribute assemblyID = "CYL-ASM-3";
    part cylinder : Cylinder;
    part piston : Piston;
    part connectingRod : ConnectingRod;
}

part fourCylinderAssembly4 : CylinderAssembly {
    attribute assemblyID = "CYL-ASM-4";
    part cylinder : Cylinder;
    part piston : Piston;
    part connectingRod : ConnectingRod;
}

part fourCylinderEngine : EngineAssembly {
    attribute engineID = "ENG-4CYL-001";
    attribute engineName = "FourCylinderEngine";
    attribute numberOfCylinders = 4;
    part crankshaft : Crankshaft;
    part cylinderAssemblies : CylinderAssembly[1..*];
}

part sixCylinderCrankshaft : Crankshaft {
    attribute crankshaftID = "CRANK-6CYL";
    attribute length = 0.60;
    attribute mass = 18.0;
    attribute material = "ForgedSteel";
    attribute numberOfJournals = 7;
}

part sixCylinderEngine : EngineAssembly {
    attribute engineID = "ENG-6CYL-001";
    attribute engineName = "SixCylinderEngine";
    attribute numberOfCylinders = 6;
    part crankshaft : Crankshaft;
    part cylinderAssemblies : CylinderAssembly[1..*];
}

part eightCylinderCrankshaft : Crankshaft {
    attribute crankshaftID = "CRANK-8CYL";
    attribute length = 0.75;
    attribute mass =