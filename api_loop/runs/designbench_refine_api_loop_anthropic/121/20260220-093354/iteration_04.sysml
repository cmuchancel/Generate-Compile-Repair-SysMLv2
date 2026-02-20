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
    attribute cylinderID : Integer;
    attribute bore : Real;
    attribute stroke : Real;
    attribute compressionRatio : Real;
    attribute displacement : Real;
}

part def Piston {
    attribute pistonID : Integer;
    attribute mass : Real;
    attribute diameter : Real;
    attribute material : String;
}

part def ConnectingRod {
    attribute rodID : Integer;
    attribute length : Real;
    attribute mass : Real;
    attribute material : String;
}

part def Crankshaft {
    attribute crankshaftID : Integer;
    attribute mass : Real;
    attribute length : Real;
    attribute material : String;
    attribute numberOfThrows : Integer;
}

part def CylinderAssembly {
    attribute assemblyID : Integer;
    part cylinder : Cylinder;
    part piston : Piston;
    part connectingRod : ConnectingRod;
}

part def Engine {
    attribute engineID : Integer;
    attribute engineType : EngineType;
    attribute numberOfCylinders : Integer;
    attribute totalDisplacement : Real;
    attribute power : Real;
    attribute torque : Real;
    attribute mass : Real;
    attribute thermalEfficiency : Real;
    attribute fuelConsumption : Real;
    attribute manufacturingCost : Real;
    attribute maintenanceCost : Real;
    attribute totalCost : Real;
    attribute reliabilityScore : Real;
    attribute meanTimeBetweenFailures : Real;
    attribute emissionLevel : Real;
    part crankshaft : Crankshaft;
}

part def FourCylinderEngine {
    attribute engineID : Integer;
    attribute engineType : EngineType;
    attribute numberOfCylinders : Integer;
    attribute totalDisplacement : Real;
    attribute power : Real;
    attribute torque : Real;
    attribute mass : Real;
    attribute thermalEfficiency : Real;
    attribute fuelConsumption : Real;
    attribute manufacturingCost : Real;
    attribute maintenanceCost : Real;
    attribute totalCost : Real;
    attribute reliabilityScore : Real;
    attribute meanTimeBetweenFailures : Real;
    attribute emissionLevel : Real;
    part cylinderAssembly1 : CylinderAssembly;
    part cylinderAssembly2 : CylinderAssembly;
    part cylinderAssembly3 : CylinderAssembly;
    part cylinderAssembly4 : CylinderAssembly;
    part crankshaft : Crankshaft;
}

part def SixCylinderEngine {
    attribute engineID : Integer;
    attribute engineType : EngineType;
    attribute numberOfCylinders : Integer;
    attribute totalDisplacement : Real;
    attribute power : Real;
    attribute torque : Real;
    attribute mass : Real;
    attribute thermalEfficiency : Real;
    attribute fuelConsumption : Real;
    attribute manufacturingCost : Real;
    attribute maintenanceCost : Real;
    attribute totalCost : Real;
    attribute reliabilityScore : Real;
    attribute meanTimeBetweenFailures : Real;
    attribute emissionLevel : Real;
    part cylinderAssembly1 : CylinderAssembly;
    part cylinderAssembly2 : CylinderAssembly;
    part cylinderAssembly3 : CylinderAssembly;
    part cylinderAssembly4 : CylinderAssembly;
    part cylinderAssembly5 : CylinderAssembly;
    part cylinderAssembly6 : CylinderAssembly;
    part crankshaft : Crankshaft;
}

part def EightCylinderEngine {
    attribute engineID : Integer;
    attribute engineType : EngineType;
    attribute numberOfCylinders : Integer;
    attribute totalDisplacement : Real;
    attribute power : Real;
    attribute torque : Real;
    attribute mass : Real;
    attribute thermalEfficiency : Real;
    attribute fuelConsumption : Real;
    attribute manufacturingCost : Real;
    attribute maintenanceCost : Real;
    attribute totalCost : Real;
    attribute reliabilityScore : Real;
    attribute meanTimeBetweenFailures : Real;
    attribute emissionLevel : Real;
    part cylinderAssembly1 : CylinderAssembly;
    part cylinderAssembly2 : CylinderAssembly;
    part cylinderAssembly3 : CylinderAssembly;
    part cylinderAssembly4 : CylinderAssembly;
    part cylinderAssembly5 : CylinderAssembly;
    part cylinderAssembly6 : CylinderAssembly;
    part cylinderAssembly7 : CylinderAssembly;
    part cylinderAssembly8 : CylinderAssembly;
    part crankshaft : Crankshaft;
}

part def PerformanceWeight {
    attribute powerWeight : Real;
    attribute massWeight : Real;
    attribute efficiencyWeight : Real;
    attribute reliabilityWeight : Real;
    attribute costWeight : Real;
}

part def EngineEvaluationScore {
    attribute evaluationID : Integer;
    attribute engineID : Integer;
    attribute engineType : EngineType;
    attribute powerScore : Real;
    attribute massScore : Real;
    attribute efficiencyScore : Real;
    attribute reliabilityScore : Real;
    attribute costScore : Real;
    attribute compositeScore : Real;
    attribute powerRating : EvaluationRating;
    attribute massRating : EvaluationRating;
    attribute efficiencyRating : EvaluationRating;
    attribute reliabilityRating : EvaluationRating;
    attribute costRating : EvaluationRating;
    attribute overallRating : EvaluationRating;
    attribute recommendationStatus : RecommendationStatus;
}

part def TradeoffAnalysisModel {
    attribute modelID : Integer;
    attribute modelName : String;
    attribute analysisDescription : String;
    attribute numberOfAlternatives : Integer;
    attribute numberOfCriteria : Integer;
    attribute bestCompositeScore : Real;
    attribute recommendedEngineID : Integer;
    attribute recommendedEngineType : EngineType;
    attribute selectionRationale : String;
    attribute decisionTransparencyNote : String;
    part weights : PerformanceWeight;
}

part def EngineRecommendation {
    attribute recommendationID : Integer;
    attribute recommendedEngineType : EngineType;
    attribute compositeScore : Real;
    attribute rationale : String;
    attribute powerRationale : String;
    attribute massRationale : String;
    attribute efficiencyRationale : String;
    attribute reliabilityRationale : String;
    attribute costRationale : String;
    attribute traceabilityNote : String;
    attribute status : RecommendationStatus;
}

port def PowerOutputPort {
    attribute power : Real;
    attribute torque : Real;
    attribute rotationalSpeed : Real;
}

port def DataOutputPort {
    attribute dataLabel : String;
    attribute dataValue : Real;
}

part def Transmission {
    attribute transmissionID : Integer;
    attribute gearRatios : Real;
    attribute efficiency : Real;
    attribute mass : Real;
    port inputPort : PowerOutputPort;
}

part def Drivetrain {
    attribute drivetrainID : Integer;
    attribute driveType : String;
    attribute efficiency : Real;
    attribute mass : Real;
    port inputPort : PowerOutputPort;
}

part def VehiclePowertrainSystem {
    attribute systemID : Integer;
    attribute vehicleModel : String;
    attribute targetPower : Real;
    attribute targetMass : Real;
    attribute targetEfficiency : Real;
    attribute targetReliability : Real;
    attribute targetCost : Real;
    attribute selectedEngineType : EngineType;
    attribute systemDescription : String;

    part fourCylEngine : FourCylinderEngine;
    part sixCylEngine : SixCylinderEngine;
    part eightCylEngine : EightCylinderEngine;
    part transmission : Transmission;
    part drivetrain : Drivetrain;
    part tradeoffModel : TradeoffAnalysisModel;
    part fourCylScore : EngineEvaluationScore;
    part sixCylScore : EngineEvaluationScore;
    part eightCylScore : EngineEvaluationScore;
    part recommendation : EngineRecommendation;

    port powerOutput : PowerOutputPort;
    port analysisOutput : DataOutputPort;
}

part vehiclePowertrain : VehiclePowertrainSystem {
    attribute systemID = 1;
    attribute vehicleModel = "StandardVehicle";
    attribute targetPower = 150.0;
    attribute targetMass = 180.0;
    attribute targetEfficiency = 0.38;
    attribute targetReliability = 0.95;
    attribute targetCost = 5000.0;
    attribute selectedEngineType = EngineType::FourCylinder;
    attribute systemDescription = "Multi-attribute trade-off analysis for optimal engine selection";

    part fourCylEngine : FourCylinderEngine {
        attribute engineID = 1;
        attribute engineType = EngineType::FourCylinder;
        attribute numberOfCylinders = 4;
        attribute totalDisplacement = 2.0;
        attribute power = 120.0;
        attribute torque = 200.0;
        attribute mass = 140.0;
        attribute thermalEfficiency = 0.36;
        attribute fuelConsumption = 7.5;
        attribute manufacturingCost = 3000.0;
        attribute maintenanceCost = 800.0;
        attribute totalCost = 3800.0;
        attribute reliabilityScore = 0.92;
        attribute meanTimeBetweenFailures = 8000.0;
        attribute emissionLevel = 120.0;

        part crankshaft : Crankshaft {
            attribute crankshaftID = 1;
            attribute mass = 12.0;
            attribute length = 0.35;
            attribute material = "ForgedSteel";
            attribute numberOfThrows = 4;
        }
        part cylinderAssembly1 : CylinderAssembly {
            attribute assemblyID = 1;
            part cylinder : Cylinder {
                attribute cylinderID = 1;
                attribute bore = 86.0;
                attribute stroke = 86.0;
                attribute compressionRatio = 10.5;
                attribute displacement = 0.5;
            }
            part piston : Piston {
                attribute pistonID = 1;
                attribute mass = 0.35;
                attribute diameter = 86.0;
                attribute material = "AluminumAlloy";
            }
            part connectingRod : ConnectingRod {
                attribute rodID = 1;
                attribute length = 0.145;
                attribute mass = 0.55;
                attribute material = "Steel";
            }
        }
        part cylinderAssembly2 : CylinderAssembly {
            attribute assemblyID = 2;
            part cylinder : Cylinder {
                attribute cylinderID =