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

part evaluationModelInst : EvaluationModel;

part optimalRecommendationInst : OptimalRecommendation;

part fourCylinderSchemeInst : EngineScheme;

part fourCylinderCrankshaft : Crankshaft;

part fourCylinderCylAsmInst : CylinderAssembly;

part fourCylinderCylInst : Cylinder;

part fourCylinderPistonInst : Piston;

part fourCylinderRodInst : ConnectingRod;

part fourCylinderPerfInst : PerformanceData;

part fourCylinderMassInst : MassData;

part fourCylinderRelInst : ReliabilityData;

part fourCylinderCostInst : CostData;

part sixCylinderSchemeInst : EngineScheme;

part sixCylinderCrankshaft : Crankshaft;

part sixCylinderCylAsmInst : CylinderAssembly;

part sixCylinderCylInst : Cylinder;

part sixCylinderPistonInst : Piston;

part sixCylinderRodInst : ConnectingRod;

part sixCylinderPerfInst : PerformanceData;

part sixCylinderMassInst : MassData;

part sixCylinderRelInst : ReliabilityData;

part sixCylinderCostInst : CostData;

part eightCylinderSchemeInst : EngineScheme;

part eightCylinderCrankshaft : Crankshaft;

part eightCylinderCylAsmInst : CylinderAssembly;

part eightCylinderCylInst : Cylinder;

part eightCylinderPistonInst : Piston;

part eightCylinderRodInst : ConnectingRod;

part eightCylinderPerfInst : PerformanceData;

part eightCylinderMassInst : MassData;

part eightCylinderRelInst : ReliabilityData;

part eightCylinderCostInst : CostData;

part hybridSchemeInst : EngineScheme;

part hybridCrankshaft : Crankshaft;

part hybridCylAsmInst : CylinderAssembly;

part hybridCylInst : Cylinder;

part hybridPistonInst : Piston;

part hybridRodInst : ConnectingRod;

part hybridPerfInst : PerformanceData;

part hybridMassInst : MassData;

part hybridRelInst : ReliabilityData;

part hybridCostInst : CostData;

part electricSchemeInst : EngineScheme;

part electricPerfInst : PerformanceData;

part electricMassInst : MassData;

part electricRelInst : ReliabilityData;

part electricCostInst : CostData;

part fourCylinderResultInst : EvaluationResult;

part fourCylinderScoresInst : NormalizedScores;

part sixCylinderResultInst : EvaluationResult;

part sixCylinderScoresInst : NormalizedScores;

part eightCylinderResultInst : EvaluationResult;

part eightCylinderScoresInst : NormalizedScores;

part hybridResultInst : EvaluationResult;

part hybridScoresInst : NormalizedScores;

part electricResultInst : EvaluationResult;

part electricScoresInst : NormalizedScores;

part tradeoffWeightsInst : TradeoffWeights;

}