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
    attribute numberOfJournals : Integer;
    attribute material : String;
}

part def CylinderAssembly {
    attribute assemblyId : Integer;
    part cylinder : Cylinder;
    part piston : Piston;
    part connectingRod : ConnectingRod;
}

part def PerformanceAttribute {
    attribute attributeName : String;
    attribute weight : Real;
    attribute normalizedScore : Real;
    attribute weightedScore : Real;
    attribute rating : EvaluationRating;
}

part def TradeoffEvaluation {
    attribute evaluationId : Integer;
    attribute engineType : EngineType;
    attribute powerScore : Real;
    attribute massScore : Real;
    attribute efficiencyScore : Real;
    attribute reliabilityScore : Real;
    attribute costScore : Real;
    attribute totalWeightedScore : Real;
    attribute recommendation : RecommendationStatus;
    attribute evaluationNotes : String;
    part powerAttribute : PerformanceAttribute;
    part massAttribute : PerformanceAttribute;
    part efficiencyAttribute : PerformanceAttribute;
    part reliabilityAttribute : PerformanceAttribute;
    part costAttribute : PerformanceAttribute;
}

part def EvaluationCriteria {
    attribute powerWeight : Real;
    attribute massWeight : Real;
    attribute efficiencyWeight : Real;
    attribute reliabilityWeight : Real;
    attribute costWeight : Real;
    attribute minPowerThreshold : Real;
    attribute maxMassThreshold : Real;
    attribute minEfficiencyThreshold : Real;
    attribute minReliabilityThreshold : Real;
    attribute maxCostThreshold : Real;
}

part def OptimizationResult {
    attribute resultId : Integer;
    attribute selectedEngineType : EngineType;
    attribute selectionRationale : String;
    attribute overallScore : Real;
    attribute rankFourCylinder : Integer;
    attribute rankSixCylinder : Integer;
    attribute rankEightCylinder : Integer;
    attribute decisionTransparencyNote : String;
    attribute traceabilityReference : String;
}

port def PowertrainPort {
    attribute maxPower : Real;
    attribute maxTorque : Real;
    attribute interfaceType : String;
}

port def EvaluationPort {
    attribute requestType : String;
    attribute responseType : String;
}

part def FourCylinderEngine {
    attribute engineId : Integer;
    attribute engineType : EngineType;
    attribute numberOfCylinders : Integer;
    attribute powerOutput : Real;
    attribute torque : Real;
    attribute totalMass : Real;
    attribute thermalEfficiency : Real;
    attribute fuelConsumption : Real;
    attribute reliabilityScore : Real;
    attribute estimatedCost : Real;
    attribute emissionsLevel : Real;
    part cylinderAssembly1 : CylinderAssembly;
    part cylinderAssembly2 : CylinderAssembly;
    part cylinderAssembly3 : CylinderAssembly;
    part cylinderAssembly4 : CylinderAssembly;
    part crankshaft : Crankshaft;
}

part def SixCylinderEngine {
    attribute engineId : Integer;
    attribute engineType : EngineType;
    attribute numberOfCylinders : Integer;
    attribute powerOutput : Real;
    attribute torque : Real;
    attribute totalMass : Real;
    attribute thermalEfficiency : Real;
    attribute fuelConsumption : Real;
    attribute reliabilityScore : Real;
    attribute estimatedCost : Real;
    attribute emissionsLevel : Real;
    part cylinderAssembly1 : CylinderAssembly;
    part cylinderAssembly2 : CylinderAssembly;
    part cylinderAssembly3 : CylinderAssembly;
    part cylinderAssembly4 : CylinderAssembly;
    part cylinderAssembly5 : CylinderAssembly;
    part cylinderAssembly6 : CylinderAssembly;
    part crankshaft : Crankshaft;
}

part def EightCylinderEngine {
    attribute engineId : Integer;
    attribute engineType : EngineType;
    attribute numberOfCylinders : Integer;
    attribute powerOutput : Real;
    attribute torque : Real;
    attribute totalMass : Real;
    attribute thermalEfficiency : Real;
    attribute fuelConsumption : Real;
    attribute reliabilityScore : Real;
    attribute estimatedCost : Real;
    attribute emissionsLevel : Real;
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

part def PowertrainConfigurationSystem {
    attribute systemId : Integer;
    attribute systemName : String;
    attribute version : String;
    attribute evaluationStatus : String;
    port powertrainInterface : PowertrainPort;
    port evaluationInterface : EvaluationPort;
    part criteria : EvaluationCriteria;
    part optimizationResult : OptimizationResult;
    part fourCylinderCandidate : FourCylinderEngine;
    part sixCylinderCandidate : SixCylinderEngine;
    part eightCylinderCandidate : EightCylinderEngine;
    part fourCylinderEvaluation : TradeoffEvaluation;
    part sixCylinderEvaluation : TradeoffEvaluation;
    part eightCylinderEvaluation : TradeoffEvaluation;
}

part def Vehicle {
    attribute vehicleId : Integer;
    attribute vehicleName : String;
    attribute targetPowerRequirement : Real;
    attribute targetMassLimit : Real;
    attribute targetEfficiencyGoal : Real;
    attribute targetBudget : Real;
    port vehiclePowertrainPort : PowertrainPort;
    part powertrainSystem : PowertrainConfigurationSystem;
}

part vehicleSystem : Vehicle {
    attribute vehicleId = 1;
    attribute vehicleName = "OptimalVehicle";
    attribute targetPowerRequirement = 150.0;
    attribute targetMassLimit = 250.0;
    attribute targetEfficiencyGoal = 0.38;
    attribute targetBudget = 5000.0;

    part powertrainSystem : PowertrainConfigurationSystem {
        attribute systemId = 1;
        attribute systemName = "PowertrainConfigSystem";
        attribute version = "1.0";
        attribute evaluationStatus = "Active";

        part criteria : EvaluationCriteria {
            attribute powerWeight = 0.25;
            attribute massWeight = 0.15;
            attribute efficiencyWeight = 0.25;
            attribute reliabilityWeight = 0.20;
            attribute costWeight = 0.15;
            attribute minPowerThreshold = 100.0;
            attribute maxMassThreshold = 300.0;
            attribute minEfficiencyThreshold = 0.30;
            attribute minReliabilityThreshold = 0.80;
            attribute maxCostThreshold = 8000.0;
        }

        part optimizationResult : OptimizationResult {
            attribute resultId = 1;
            attribute selectedEngineType = EngineType::FourCylinder;
            attribute selectionRationale = "FourCylinder achieves highest weighted score of 0.889 balancing efficiency cost and reliability";
            attribute overallScore = 0.889;
            attribute rankFourCylinder = 1;
            attribute rankSixCylinder = 2;
            attribute rankEightCylinder = 3;
            attribute decisionTransparencyNote = "Weighted multi-attribute scoring applied across power mass efficiency reliability cost dimensions";
            attribute traceabilityReference = "TradeoffEvaluation IDs 1 2 3 support this recommendation";
        }

        part fourCylinderCandidate : FourCylinderEngine {
            attribute engineId = 101;
            attribute engineType = EngineType::FourCylinder;
            attribute numberOfCylinders = 4;
            attribute powerOutput = 120.0;
            attribute torque = 200.0;
            attribute totalMass = 150.0;
            attribute thermalEfficiency = 0.38;
            attribute fuelConsumption = 6.5;
            attribute reliabilityScore = 0.90;
            attribute estimatedCost = 3500.0;
            attribute emissionsLevel = 120.0;

            part cylinderAssembly1 : CylinderAssembly {
                attribute assemblyId = 1;
                part cylinder : Cylinder {
                    attribute cylinderId = 1;
                    attribute bore = 82.0;
                    attribute stroke = 90.0;
                    attribute displacement = 0.476;
                    attribute compressionRatio = 10.5;
                }
                part piston : Piston {
                    attribute pistonId = 1;
                    attribute diameter = 82.0;
                    attribute mass = 0.45;
                    attribute material = "AluminumAlloy";
                }
                part connectingRod : ConnectingRod {
                    attribute rodId = 1;
                    attribute length = 145.0;
                    attribute mass = 0.55;
                    attribute material = "SteelAlloy";
                }
            }

            part cylinderAssembly2 : CylinderAssembly {
                attribute assemblyId = 2;
                part cylinder : Cylinder {
                    attribute cylinderId = 2;
                    attribute bore = 82.0;
                    attribute stroke = 90.0;
                    attribute displacement = 0.476;
                    attribute compressionRatio = 10.5;
                }
                part piston : Piston {
                    attribute pistonId = 2;
                    attribute diameter = 82.0;
                    attribute mass = 0.45;
                    attribute material = "AluminumAlloy";
                }
                part connectingRod : ConnectingRod {
                    attribute rodId = 2;
                    attribute length = 145.0;
                    attribute mass = 0.55;
                    attribute material = "SteelAlloy";
                }
            }

            part cylinderAssembly3 : CylinderAssembly {
                attribute assemblyId = 3;
                part cylinder : Cylinder {
                    attribute cylinderId = 3;
                    attribute bore = 82.0;
                    attribute stroke = 90.0;
                    attribute displacement = 0.476;
                    attribute compressionRatio = 10.5;
                }
                part piston : Piston {
                    attribute pistonId = 3;
                    attribute diameter = 82.0;
                    attribute mass = 0.45;
                    attribute material = "AluminumAlloy";
                }
                part connectingRod : ConnectingRod {
                    attribute rodId = 3;
                    attribute length = 145.0;
                    attribute mass = 0.55;
                    attribute material = "SteelAlloy";
                }
            }

            part cylinderAssembly4 : CylinderAssembly {
                attribute assemblyId = 4;
                part cylinder : Cylinder {
                    attribute cylinderId = 4;
                    attribute bore = 82.0;
                    attribute stroke = 90.0;
                    attribute displacement = 0.476;
                    attribute compressionRatio = 10.5;
                }
                part piston : Piston {
                    attribute pistonId = 4;
                    attribute diameter = 82.0;
                    attribute mass = 0.45;
                    attribute material = "AluminumAlloy";
                }
                part connectingRod : ConnectingRod {
                    attribute rodId = 4;
                    attribute length = 145.0;
                    attribute mass = 0.55;
                    attribute material = "SteelAlloy";
                }
            }

            part crankshaft : Crankshaft {
                attribute crankshaftId = 1;
                attribute length = 450.0;
                attribute mass = 12.0;
                attribute numberOfJournals = 5;
                attribute material = "ForgedSteel";
            }
        }

        part sixCylinderCandidate : SixCylinderEngine {
            attribute engineId = 102;
            attribute engineType = EngineType::SixCylinder;
            attribute numberOfCylinders = 6;
            attribute powerOutput = 180.0;
            attribute torque = 320.0;
            attribute totalMass = 210.0;
            attribute thermalEfficiency = 0.36;
            attribute fuelConsumption = 8.5;
            attribute reliabilityScore = 0.88;
            attribute estimatedCost = 5500.0;
            attribute emissionsLevel = 150.0;

            part cylinderAssembly1 : CylinderAssembly {
                attribute assemblyId = 1;
                part cylinder : Cylinder {
                    attribute cylinderId = 1;
                    attribute bore = 86.0;
                    attribute stroke = 86.0;
                    attribute displacement = 0.499;
                    attribute compressionRatio = 11.0;
                }
                part piston : Piston {
                    attribute pistonId = 1;
                    attribute diameter = 86.0;
                    attribute mass = 0.48;
                    attribute material = "AluminumAlloy";
                }
                part connectingRod : ConnectingRod {
                    attribute rodId = 1;
                    attribute length = 152.0;
                    attribute mass = 0.60;
                    attribute material = "SteelAlloy";
                }
            }

            part cylinderAssembly2 : CylinderAssembly {
                attribute assemblyId = 2;
                part cylinder : Cylinder {
                    attribute cylinderId = 2;
                    attribute bore = 86.0;
                    attribute stroke = 86.0;
                    attribute displacement = 0.499;
                    attribute compressionRatio = 11.0;
                }
                part piston : Piston {
                    attribute pistonId = 2;
                    attribute diameter = 86.0;
                    attribute mass = 0.48;
                    attribute material = "AluminumAlloy";
                }
                part connectingRod : ConnectingRod {
                    attribute rodId = 2;
                    attribute length = 152.0;
                    attribute mass = 0.60;
                    attribute material = "SteelAlloy";
                }
            }

            part cylinderAssembly3 : CylinderAssembly {
                attribute assemblyId = 3;
                part cylinder : Cylinder {
                    attribute cylinderId = 3;
                    attribute bore = 86.0;
                    attribute stroke = 86.0;
                    attribute displacement = 0.499;
                    attribute compressionRatio = 11.0;
                }
                part piston : Piston {
                    attribute pistonId = 3;
                    attribute diameter = 86.0;
                    attribute mass = 0.48;
                    attribute material = "AluminumAlloy";
                }
                part connectingRod : ConnectingRod {
                    attribute rodId = 3;
                    attribute length = 152.0;
                    attribute mass = 0.60;
                    attribute material = "SteelAlloy";
                }
            }

            part cylinderAssembly4 : CylinderAssembly {
                attribute assemblyId = 4;
                part cylinder : Cylinder {
                    attribute cylinderId = 4;
                    attribute bore = 86.0;
                    attribute stroke = 86.0;
                    attribute displacement = 0.499;
                    attribute compressionRatio = 11.0;
                }
                part piston : Piston {
                    attribute pistonId = 4;
                    attribute diameter = 86.0;
                    attribute mass = 0.48;
                    attribute material = "AluminumAlloy";
                }
                part connectingRod : ConnectingRod {
                    attribute rodId = 4;
                    attribute length = 152.0;
                    attribute mass = 0.60;
                    attribute material = "SteelAlloy";
                }
            }

            part cylinderAssembly5 : CylinderAssembly {
                attribute assemblyId = 5;
                part cylinder : Cylinder {
                    attribute cylinderId = 5;
                    attribute bore = 86.0;
                    attribute stroke = 86.0;
                    attribute displacement = 0.499;
                    attribute compressionRatio = 11.0;
                }
                part piston : Piston {
                    attribute pistonId = 5;
                    attribute diameter = 86.0;
                    attribute mass = 0.48;
                    attribute material = "AluminumAlloy";
                }
                part connectingRod : ConnectingRod {
                    attribute rodId = 5;
                    attribute length = 152.0;
                    attribute mass = 0.60;
                    attribute material = "SteelAlloy";
                }
            }

            part cylinderAssembly6 : CylinderAssembly {
                attribute assemblyId = 6;
                part cylinder : Cylinder {
                    attribute cylinderId = 6;
                    attribute bore = 86.0;
                    attribute stroke = 86.0;
                    attribute displacement = 0.499;
                    attribute compressionRatio = 11.0;
                }
                part piston : Piston {
                    attribute pistonId = 6;
                    attribute diameter = 86.0;
                    attribute mass = 0.48;
                    attribute material = "AluminumAlloy";
                }
                part connectingRod : ConnectingRod {
                    attribute rodId = 6;
                    attribute length = 152.0;
                    attribute mass = 0.60;
                    attribute material = "SteelAlloy";
                }
            }

            part crankshaft : Crankshaft {
                attribute crankshaftId = 2;
                attribute length = 620.0;
                attribute mass = 18.0;
                attribute numberOfJournals = 7;
                attribute material = "ForgedSteel";
            }
        }

        part eightCylinderCandidate : EightCylinderEngine {
            attribute engineId = 103;
            attribute engineType = EngineType::EightCylinder;
            attribute numberOfCylinders = 8;
            attribute powerOutput = 280.0;
            attribute torque = 480.0;
            attribute totalMass = 290.0;
            attribute thermalEfficiency = 0.33;
            attribute fuelConsumption = 12.0;
            attribute reliabilityScore = 0.85;
            attribute estimatedCost = 8000.0;
            attribute emissionsLevel = 210.0;

            part cylinderAssembly1 : CylinderAssembly {
                attribute assemblyId = 1;
                part cylinder : Cylinder {
                    attribute cylinderId = 1;
                    attribute bore = 92.0;
                    attribute stroke = 84.0;
                    attribute displacement = 0.560;
                    attribute compressionRatio = 10.0;
                }
                part piston : Piston {
                    attribute pistonId = 1;
                    attribute diameter = 92.0;
                    attribute mass = 0.52;
                    attribute material = "AluminumAlloy";
                }
                part connectingRod : ConnectingRod {
                    attribute rodId = 1;
                    attribute length = 158.0;
                    attribute mass = 0.65;
                    attribute material = "SteelAlloy";
                }
            }

            part cylinderAssembly2 : CylinderAssembly {
                attribute assemblyId = 2;
                part cylinder : Cylinder {
                    attribute cylinderId = 2;
                    attribute bore = 92.0;
                    attribute stroke = 84.0;
                    attribute displacement = 0.560;
                    attribute compressionRatio = 10.0;
                }
                part piston : Piston {
                    attribute pistonId = 2;
                    attribute diameter = 92.0;
                    attribute mass = 0.52;
                    attribute material = "AluminumAlloy";
                }
                part connectingRod : ConnectingRod {
                    attribute rodId = 2;
                    attribute length = 158.0;
                    attribute mass = 0.65;
                    attribute material = "SteelAlloy";
                }
            }

            part cylinderAssembly3 : CylinderAssembly {
                attribute assemblyId = 3;
                part cylinder : Cylinder {
                    attribute cylinderId = 3;
                    attribute bore = 92.0;
                    attribute stroke = 84.0;
                    attribute displacement = 0.560;
                    attribute compressionRatio = 10.0;
                }
                part piston : Piston {
                    attribute pistonId = 3;
                    attribute diameter = 92.0;
                    attribute mass = 0.52;
                    attribute material = "AluminumAlloy";
                }
                part connectingRod : ConnectingRod {
                    attribute rodId = 3;
                    attribute length = 158.0;
                    attribute mass = 0.65;
                    attribute material = "SteelAlloy";
                }
            }

            part cylinderAssembly4 : CylinderAssembly {
                attribute assemblyId = 4;
                part cylinder : Cylinder {
                    attribute cylinderId = 4;
                    attribute bore = 92.0;
                    attribute stroke = 84.0;
                    attribute displacement = 0.560;
                    attribute compressionRatio = 10.0;
                }
                part piston : Piston {
                    attribute pistonId = 4;
                    attribute diameter = 92.0;
                    attribute mass = 0.52;
                    attribute material = "AluminumAlloy";
                }
                part connectingRod : ConnectingRod {
                    attribute rodId = 4;
                    attribute length = 158.0;
                    attribute mass = 0.65;
                    attribute material = "SteelAlloy";
                }
            }

            part cylinderAssembly5 : CylinderAssembly {
                attribute assemblyId = 5;
                part cylinder : Cylinder {
                    attribute cylinderId = 5;
                    attribute bore = 92.0;
                    attribute stroke = 84.0;
                    attribute displacement = 0.560;
                    attribute compressionRatio = 10.0;
                }
                part piston : Piston {
                    attribute pistonId = 5;
                    attribute diameter = 92.0;
                    attribute mass = 0.52;
                    attribute material = "AluminumAlloy";
                }
                part connectingRod : ConnectingRod {
                    attribute rodId = 5;
                    attribute length = 158.0;
                    attribute mass = 0.65;
                    attribute material = "SteelAlloy";
                }
            }

            part cylinderAssembly6 : CylinderAssembly {
                attribute assemblyId = 6;
                part cylinder : Cylinder {
                    attribute cylinderId = 6;
                    attribute bore = 92.0;
                    attribute stroke = 84.0;
                    attribute displacement = 0.560;
                    attribute compressionRatio = 10.0;
                }
                part piston : Piston {
                    attribute pistonId = 6;
                    attribute diameter = 92.0;
                    attribute mass = 0.52;
                    attribute material = "AluminumAlloy";
                }
                part connectingRod : ConnectingRod {
                    attribute rodId = 6;
                    attribute length = 158.0;
                    attribute mass = 0.65;
                    attribute material = "SteelAlloy";
                }
            }

            part cylinderAssembly7 : CylinderAssembly {
                attribute assemblyId = 7;
                part cylinder : Cylinder {
                    attribute cylinderId = 7;
                    attribute bore = 92.0;
                    attribute stroke = 84.0;
                    attribute displacement = 0.560;
                    attribute compressionRatio = 10.0;
                }
                part piston : Piston {
                    attribute pistonId = 7;
                    attribute diameter = 92.0;
                    attribute mass = 0.52;
                    attribute material = "AluminumAlloy";
                }
                part connectingRod : ConnectingRod {
                    attribute rodId = 7;
                    attribute length = 158.0;
                    attribute mass = 0.65;
                    attribute material = "SteelAlloy";
                }
            }

            part cylinderAssembly8 : CylinderAssembly {
                attribute assemblyId = 8;
                part cylinder : Cylinder {
                    attribute cylinderId = 8;
                    attribute bore = 92.0;
                    attribute stroke = 84.0;
                    attribute displacement = 0.560;
                    attribute compressionRatio = 10.0;
                }
                part piston : Piston {
                    attribute pistonId = 8;
                    attribute diameter = 92.0;
                    attribute mass = 0.52;
                    attribute material = "AluminumAlloy";
                }
                part connectingRod : ConnectingRod {
                    attribute rodId = 8;
                    attribute length = 158.0;
                    attribute mass = 0.65;
                    attribute material = "SteelAlloy";
                }
            }

            part crankshaft : Crankshaft {
                attribute crankshaftId = 3;
                attribute length = 780.0;
                attribute mass = 26.0;
                attribute numberOfJournals = 9;
                attribute material = "ForgedSteel";
            }
        }

        part fourCylinderEvaluation : TradeoffEvaluation {
            attribute evaluationId = 1;
            attribute engineType = EngineType::FourCylinder;
            attribute powerScore = 0.75;
            attribute massScore = 0.95;
            attribute efficiencyScore = 0.95;
            attribute reliabilityScore = 0.90;
            attribute costScore = 0.95;
            attribute totalWeightedScore = 0.889;
            attribute recommendation = RecommendationStatus::Recommended;
            attribute evaluationNotes = "FourCylinder excels in efficiency and cost with acceptable power for target requirements";

            part powerAttribute : PerformanceAttribute {
                attribute attributeName = "Power";
                attribute weight = 0.25;
                attribute normalizedScore = 0.75;
                attribute weightedScore = 0.1875;
                attribute rating = EvaluationRating::Good;
            }
            part massAttribute : PerformanceAttribute {
                attribute attributeName = "Mass";
                attribute weight = 0.15;
                attribute normalizedScore = 0.95;
                attribute weightedScore = 0.1425;
                attribute rating = EvaluationRating::Excellent;
            }
            part efficiencyAttribute : PerformanceAttribute {
                attribute attributeName = "Efficiency";
                attribute weight = 0.25;
                attribute normalizedScore = 0.95;
                attribute weightedScore = 0.2375;
                attribute rating = EvaluationRating::Excellent;
            }
            part reliabilityAttribute : PerformanceAttribute {
                attribute attributeName = "Reliability";
                attribute weight = 0.20;
                attribute normalizedScore = 0.90;
                attribute weightedScore = 0.18;
                attribute rating = EvaluationRating::Excellent;
            }
            part costAttribute : PerformanceAttribute {
                attribute attributeName = "Cost";
                attribute weight = 0.15;
                attribute normalizedScore = 0.95;
                attribute weightedScore = 0.1425;
                attribute rating = EvaluationRating::Excellent;
            }
        }

        part sixCylinderEvaluation : TradeoffEvaluation {
            attribute evaluationId = 2;
            attribute engineType = EngineType::SixCylinder;
            attribute powerScore = 0.90;
            attribute massScore = 0.75;
            attribute efficiencyScore = 0.80;
            attribute reliabilityScore = 0.88;
            attribute costScore = 0.70;
            attribute totalWeightedScore = 0.822;
            attribute recommendation = RecommendationStatus::NotRecommended;
            attribute evaluationNotes = "SixCylinder offers good power but higher mass and cost reduce overall score";

            part powerAttribute : PerformanceAttribute {
                attribute attributeName = "Power";
                attribute weight = 0.25;
                attribute normalizedScore = 0.90;
                attribute weightedScore = 0.225;
                attribute rating = EvaluationRating::Excellent;
            }
            part massAttribute : PerformanceAttribute {
                attribute attributeName = "Mass";
                attribute weight = 0.15;
                attribute normalizedScore = 0.75;
                attribute weightedScore = 0.1125;
                attribute rating = EvaluationRating::Good;
            }
            part efficiencyAttribute : PerformanceAttribute {
                attribute attributeName = "Efficiency";
                attribute weight = 0.25;
                attribute normalizedScore = 0.80;
                attribute weightedScore = 0.20;
                attribute rating = EvaluationRating::Good;
            }
            part reliabilityAttribute : PerformanceAttribute {
                attribute attributeName = "Reliability";
                attribute weight = 0.20;
                attribute normalizedScore = 0.88;
                attribute weightedScore = 0.176;
                attribute rating = EvaluationRating::Good;
            }
            part costAttribute : PerformanceAttribute {
                attribute attributeName = "Cost";
                attribute weight = 0.15;
                attribute normalizedScore = 0.70;
                attribute weightedScore = 0.105;
                attribute rating = EvaluationRating::Good;
            }
        }

        part eightCylinderEvaluation : TradeoffEvaluation {
            attribute evaluationId = 3;