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
    part cylinderAssemblies : CylinderAssembly[4];
}

part def SixCylinderEngine {
    attribute engineId : Integer;
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
    part cylinderAssemblies : CylinderAssembly[6];
}

part def EightCylinderEngine {
    attribute engineId : Integer;
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
    attribute powerScore : Real;
    attribute massScore : Real;
    attribute efficiencyScore : Real;
    attribute reliabilityScore : Real;
    attribute costScore : Real;
    attribute totalWeightedScore : Real;
    attribute rankingPosition : Integer;
}

part def SelectionExplanation {
    attribute explanationId : Integer;
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
    attribute overallScore : Real;
    attribute powerScore : Real;
    attribute massScore : Real;
    attribute efficiencyScore : Real;
    attribute reliabilityScore : Real;
    attribute costScore : Real;
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
    attribute finalScore : Real;
    attribute reportSummary : String;
}

part def FourCylinderCrankshaft {
    attribute crankshaftId : Integer;
    attribute length : Real;
    attribute mass : Real;
    attribute material : String;
    attribute numberOfThrows : Integer;
}

part def SixCylinderCrankshaft {
    attribute crankshaftId : Integer;
    attribute length : Real;
    attribute mass : Real;
    attribute material : String;
    attribute numberOfThrows : Integer;
}

part def EightCylinderCrankshaft {
    attribute crankshaftId : Integer;
    attribute length : Real;
    attribute mass : Real;
    attribute material : String;
    attribute numberOfThrows : Integer;
}

part def FourCylinderCylinderAssembly {
    attribute assemblyId : Integer;
    part cylinder : Cylinder;
    part piston : Piston;
    part connectingRod : ConnectingRod;
}

part def SixCylinderCylinderAssembly {
    attribute assemblyId : Integer;
    part cylinder : Cylinder;
    part piston : Piston;
    part connectingRod : ConnectingRod;
}

part def EightCylinderCylinderAssembly {
    attribute assemblyId : Integer;
    part cylinder : Cylinder;
    part piston : Piston;
    part connectingRod : ConnectingRod;
}

part def FourCylinderEngineInstance {
    attribute engineId : Integer;
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
    part crankshaft : FourCylinderCrankshaft;
    part cyl1 : FourCylinderCylinderAssembly;
    part cyl2 : FourCylinderCylinderAssembly;
    part cyl3 : FourCylinderCylinderAssembly;
    part cyl4 : FourCylinderCylinderAssembly;
}

part def SixCylinderEngineInstance {
    attribute engineId : Integer;
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
    part crankshaft : SixCylinderCrankshaft;
    part cyl1 : SixCylinderCylinderAssembly;
    part cyl2 : SixCylinderCylinderAssembly;
    part cyl3 : SixCylinderCylinderAssembly;
    part cyl4 : SixCylinderCylinderAssembly;
    part cyl5 : SixCylinderCylinderAssembly;
    part cyl6 : SixCylinderCylinderAssembly;
}

part def EightCylinderEngineInstance {
    attribute engineId : Integer;
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
    part crankshaft : EightCylinderCrankshaft;
    part cyl1 : EightCylinderCylinderAssembly;
    part cyl2 : EightCylinderCylinderAssembly;
    part