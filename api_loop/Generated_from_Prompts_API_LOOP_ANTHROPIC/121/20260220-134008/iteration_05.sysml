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
    attribute cylinderId : ScalarValues::Integer;
    attribute bore : ScalarValues::Real;
    attribute stroke : ScalarValues::Real;
    attribute displacement : ScalarValues::Real;
    attribute compressionRatio : ScalarValues::Real;
}

part def Piston {
    attribute pistonId : ScalarValues::Integer;
    attribute mass : ScalarValues::Real;
    attribute diameter : ScalarValues::Real;
    attribute material : ScalarValues::String;
}

part def ConnectingRod {
    attribute rodId : ScalarValues::Integer;
    attribute length : ScalarValues::Real;
    attribute mass : ScalarValues::Real;
    attribute material : ScalarValues::String;
}

part def Crankshaft {
    attribute crankshaftId : ScalarValues::Integer;
    attribute length : ScalarValues::Real;
    attribute mass : ScalarValues::Real;
    attribute material : ScalarValues::String;
    attribute numberOfJournals : ScalarValues::Integer;
}

part def CylinderAssembly {
    attribute assemblyId : ScalarValues::Integer;
    part cylinder : Cylinder;
    part piston : Piston;
    part connectingRod : ConnectingRod;
}

part def Engine {
    attribute engineId : ScalarValues::Integer;
    attribute engineType : EngineType;
    attribute numberOfCylinders : ScalarValues::Integer;
    attribute totalDisplacement : ScalarValues::Real;
    attribute maxPower : ScalarValues::Real;
    attribute maxTorque : ScalarValues::Real;
    attribute engineMass : ScalarValues::Real;
    attribute thermalEfficiency : ScalarValues::Real;
    attribute reliabilityScore : ScalarValues::Real;
    attribute manufacturingCost : ScalarValues::Real;
    attribute fuelConsumption : ScalarValues::Real;
    part crankshaft : Crankshaft;
}

part def FourCylinderEngine {
    attribute engineId : ScalarValues::Integer;
    attribute engineType : EngineType;
    attribute numberOfCylinders : ScalarValues::Integer;
    attribute totalDisplacement : ScalarValues::Real;
    attribute maxPower : ScalarValues::Real;
    attribute maxTorque : ScalarValues::Real;
    attribute engineMass : ScalarValues::Real;
    attribute thermalEfficiency : ScalarValues::Real;
    attribute reliabilityScore : ScalarValues::Real;
    attribute manufacturingCost : ScalarValues::Real;
    attribute fuelConsumption : ScalarValues::Real;
    part crankshaft : Crankshaft;
    part assembly1 : CylinderAssembly;
    part assembly2 : CylinderAssembly;
    part assembly3 : CylinderAssembly;
    part assembly4 : CylinderAssembly;
}

part def SixCylinderEngine {
    attribute engineId : ScalarValues::Integer;
    attribute engineType : EngineType;
    attribute numberOfCylinders : ScalarValues::Integer;
    attribute totalDisplacement : ScalarValues::Real;
    attribute maxPower : ScalarValues::Real;
    attribute maxTorque : ScalarValues::Real;
    attribute engineMass : ScalarValues::Real;
    attribute thermalEfficiency : ScalarValues::Real;
    attribute reliabilityScore : ScalarValues::Real;
    attribute manufacturingCost : ScalarValues::Real;
    attribute fuelConsumption : ScalarValues::Real;
    part crankshaft : Crankshaft;
    part assembly1 : CylinderAssembly;
    part assembly2 : CylinderAssembly;
    part assembly3 : CylinderAssembly;
    part assembly4 : CylinderAssembly;
    part assembly5 : CylinderAssembly;
    part assembly6 : CylinderAssembly;
}

part def EightCylinderEngine {
    attribute engineId : ScalarValues::Integer;
    attribute engineType : EngineType;
    attribute numberOfCylinders : ScalarValues::Integer;
    attribute totalDisplacement : ScalarValues::Real;
    attribute maxPower : ScalarValues::Real;
    attribute maxTorque : ScalarValues::Real;
    attribute engineMass : ScalarValues::Real;
    attribute thermalEfficiency : ScalarValues::Real;
    attribute reliabilityScore : ScalarValues::Real;
    attribute manufacturingCost : ScalarValues::Real;
    attribute fuelConsumption : ScalarValues::Real;
    part crankshaft : Crankshaft;
    part assembly1 : CylinderAssembly;
    part assembly2 : CylinderAssembly;
    part assembly3 : CylinderAssembly;
    part assembly4 : CylinderAssembly;
    part assembly5 : CylinderAssembly;
    part assembly6 : CylinderAssembly;
    part assembly7 : CylinderAssembly;
    part assembly8 : CylinderAssembly;
}

part def PerformanceMetrics {
    attribute powerWeight : ScalarValues::Real;
    attribute efficiencyWeight : ScalarValues::Real;
    attribute reliabilityWeight : ScalarValues::Real;
    attribute costWeight : ScalarValues::Real;
    attribute massWeight : ScalarValues::Real;
    attribute powerScore : ScalarValues::Real;
    attribute efficiencyScore : ScalarValues::Real;
    attribute reliabilityScore : ScalarValues::Real;
    attribute costScore : ScalarValues::Real;
    attribute massScore : ScalarValues::Real;
    attribute compositeScore : ScalarValues::Real;
    attribute powerRating : EvaluationRating;
    attribute efficiencyRating : EvaluationRating;
    attribute reliabilityRating : EvaluationRating;
    attribute costRating : EvaluationRating;
    attribute overallRating : EvaluationRating;
}

part def TradeoffAnalysis {
    attribute analysisId : ScalarValues::Integer;
    attribute analysisName : ScalarValues::String;
    attribute numberOfCandidates : ScalarValues::Integer;
    attribute bestCandidateId : ScalarValues::Integer;
    attribute bestCompositeScore : ScalarValues::Real;
    attribute recommendation : RecommendationStatus;
    attribute selectionRationale : ScalarValues::String;
    attribute transparencyNote : ScalarValues::String;
}

part def EngineCandidate {
    attribute candidateId : ScalarValues::Integer;
    attribute candidateName : ScalarValues::String;
    attribute compositeScore : ScalarValues::Real;
    attribute recommendation : RecommendationStatus;
    attribute rationaleExplanation : ScalarValues::String;
    part metrics : PerformanceMetrics;
}

port def PowerOutputPort {
    attribute power : ScalarValues::Real;
    attribute torque : ScalarValues::Real;
    attribute speed : ScalarValues::Real;
}

port def FuelInputPort {
    attribute fuelFlowRate : ScalarValues::Real;
    attribute fuelType : ScalarValues::String;
}

port def DataPort {
    attribute dataLabel : ScalarValues::String;
    attribute dataValue : ScalarValues::Real;
}

part def Powertrain {
    attribute powertrainId : ScalarValues::Integer;
    attribute vehicleClass : ScalarValues::String;
    attribute targetPower : ScalarValues::Real;
    attribute targetEfficiency : ScalarValues::Real;
    attribute targetReliability : ScalarValues::Real;
    attribute targetCost : ScalarValues::Real;
    attribute targetMass : ScalarValues::Real;
    port powerOut : PowerOutputPort;
    port fuelIn : FuelInputPort;
    port dataOut : DataPort;
    part selectedEngine : Engine;
    part tradeoff : TradeoffAnalysis;
}

part def VehiclePowertrainSystem {
    attribute systemId : ScalarValues::Integer;
    attribute systemName : ScalarValues::String;
    attribute systemVersion : ScalarValues::String;
    attribute optimizationEnabled : ScalarValues::Boolean;
    attribute traceabilityEnabled : ScalarValues::Boolean;
    port systemDataOut : DataPort;
    part powertrain : Powertrain;
    part candidate4Cyl : EngineCandidate;
    part candidate6Cyl : EngineCandidate;
    part candidate8Cyl : EngineCandidate;
    part fourCylEngine : FourCylinderEngine;
    part sixCylEngine : SixCylinderEngine;
    part eightCylEngine : EightCylinderEngine;
    part systemTradeoff : TradeoffAnalysis;
}

part vehiclePowertrainConfigurationSystem : VehiclePowertrainSystem;

part mainPowertrain : Powertrain;

part mainEngine : Engine {
    attribute engineId = 201;
    attribute engineType = EngineType::FourCylinder;
    attribute numberOfCylinders = 4;
    attribute totalDisplacement = 2.0;
    attribute maxPower = 155.0;
    attribute maxTorque = 280.0;
    attribute engineMass = 140.0;
    attribute thermalEfficiency = 0.39;
    attribute reliabilityScore = 0.96;
    attribute manufacturingCost = 4500.0;
    attribute fuelConsumption = 7.2;
}

part mainCrankshaft : Crankshaft {
    attribute crankshaftId = 301;
    attribute length = 0.45;
    attribute mass = 12.0;
    attribute material = "Forged Steel";
    attribute numberOfJournals = 5;
}

part mainTradeoff : TradeoffAnalysis {
    attribute analysisId = 401;
    attribute analysisName = "Powertrain Engine Selection Analysis";
    attribute numberOfCandidates = 3;
    attribute bestCandidateId = 1;
    attribute bestCompositeScore = 0.87;
    attribute recommendation = RecommendationStatus::Recommended;
    attribute selectionRationale = "Four-cylinder engine achieves best balance of power efficiency and cost";
    attribute transparencyNote = "Composite score computed using weighted sum of normalized KPI scores";
}

part candidate4Cyl : EngineCandidate {
    attribute candidateId = 1;
    attribute candidateName = "Four Cylinder 2.0L";
    attribute compositeScore = 0.87;
    attribute recommendation = RecommendationStatus::Recommended;
    attribute rationaleExplanation = "Optimal balance of fuel efficiency cost and adequate power output for passenger car segment";
    part metrics : PerformanceMetrics {
        attribute powerWeight = 0.25;
        attribute efficiencyWeight = 0.30;
        attribute reliabilityWeight = 0.20;
        attribute costWeight = 0.15;
        attribute massWeight = 0.10;
        attribute powerScore = 0.78;
        attribute efficiencyScore = 0.92;
        attribute reliabilityScore = 0.96;
        attribute costScore = 0.90;
        attribute massScore = 0.88;
        attribute compositeScore = 0.87;
        attribute powerRating = EvaluationRating::Good;
        attribute efficiencyRating = EvaluationRating::Excellent;
        attribute reliabilityRating = EvaluationRating::Excellent;
        attribute costRating = EvaluationRating::Excellent;
        attribute overallRating = EvaluationRating::Excellent;
    }
}

part candidate6Cyl : EngineCandidate {
    attribute candidateId = 2;
    attribute candidateName = "Six Cylinder 3.0L";
    attribute compositeScore = 0.74;
    attribute recommendation = RecommendationStatus::NotRecommended;
    attribute rationaleExplanation = "Higher power but lower fuel efficiency and higher cost reduce overall composite score";
    part metrics : PerformanceMetrics {
        attribute powerWeight = 0.25;
        attribute efficiencyWeight = 0.30;
        attribute reliabilityWeight = 0.20;
        attribute costWeight = 0.15;
        attribute massWeight = 0.10;
        attribute powerScore = 0.92;
        attribute efficiencyScore = 0.72;
        attribute reliabilityScore = 0.90;
        attribute costScore = 0.65;
        attribute massScore = 0.70;
        attribute compositeScore = 0.74;
        attribute powerRating = EvaluationRating::Excellent;
        attribute efficiencyRating = EvaluationRating::Good;
        attribute reliabilityRating = EvaluationRating::Excellent;
        attribute costRating = EvaluationRating::Fair;
        attribute overallRating = EvaluationRating::Good;
    }
}

part candidate8Cyl : EngineCandidate {
    attribute candidateId = 3;
    attribute candidateName = "Eight Cylinder 5.0L";
    attribute compositeScore = 0.58;
    attribute recommendation = RecommendationStatus::NotRecommended;
    attribute rationaleExplanation = "Highest power but poor fuel efficiency and high cost make it unsuitable for standard passenger car";
    part metrics : PerformanceMetrics {
        attribute powerWeight = 0.25;
        attribute efficiencyWeight = 0.30;
        attribute reliabilityWeight = 0.20;
        attribute costWeight = 0.15;
        attribute massWeight = 0.10;
        attribute powerScore = 0.98;
        attribute efficiencyScore = 0.45;
        attribute reliabilityScore = 0.85;
        attribute costScore = 0.40;
        attribute massScore = 0.45;
        attribute compositeScore = 0.58;
        attribute powerRating = EvaluationRating::Excellent;
        attribute efficiencyRating = EvaluationRating::Poor;
        attribute reliabilityRating = EvaluationRating::Good;
        attribute costRating = EvaluationRating::Poor;
        attribute overallRating = EvaluationRating::Fair;
    }
}

part fourCylEngineInstance : FourCylinderEngine {
    attribute engineId = 501;
    attribute engineType = EngineType::FourCylinder;
    attribute numberOfCylinders = 4;
    attribute totalDisplacement = 2.0;
    attribute maxPower = 155.0;
    attribute maxTorque = 280.0;
    attribute engineMass = 140.0;
    attribute thermalEfficiency = 0.39;
    attribute reliabilityScore = 0.96;
    attribute manufacturingCost = 4500.0;
    attribute fuelConsumption = 7.2;
    part crankshaft : Crankshaft {
        attribute crankshaftId = 601;
        attribute length = 0.45;
        attribute mass = 12.0;
        attribute material = "Forged Steel";
        attribute numberOfJournals = 5;
    }
    part assembly1 : CylinderAssembly {
        attribute assemblyId = 701;
        part cylinder : Cylinder {
            attribute cylinderId = 801;
            attribute bore = 0.086;
            attribute stroke = 0.086;
            attribute displacement = 0.5;
            attribute compressionRatio = 10.5;
        }
        part piston : Piston {
            attribute pistonId = 901;
            attribute mass = 0.35;
            attribute diameter = 0.086;
            attribute material = "Aluminum Alloy";
        }
        part connectingRod : ConnectingRod {
            attribute rodId = 1001;
            attribute length = 0.145;
            attribute mass = 0.55;
            attribute material = "Forged Steel";
        }
    }
    part assembly2 : CylinderAssembly {
        attribute assemblyId = 702;
        part cylinder : Cylinder {
            attribute cylinderId = 802;
            attribute bore = 0.086;
            attribute stroke = 0.086;
            attribute displacement = 0.5;
            attribute compressionRatio = 10.5;
        }
        part piston : Piston {
            attribute pistonId = 902;
            attribute mass = 0.35;
            attribute diameter = 0.086;
            attribute material = "Aluminum Alloy";
        }
        part connectingRod : ConnectingRod {
            attribute rodId = 1002;
            attribute length = 0.145;
            attribute mass = 0.55;
            attribute material = "Forged Steel";
        }
    }
    part assembly3 : CylinderAssembly {
        attribute assemblyId = 703;
        part cylinder : Cylinder {
            attribute cylinderId = 803;
            attribute bore = 0.086;
            attribute stroke = 0.086;
            attribute displacement = 0.5;
            attribute compressionRatio = 10.5;
        }
        part piston : Piston {
            attribute pistonId = 903;
            attribute mass = 0.35;
            attribute diameter = 0.086;
            attribute material = "Aluminum Alloy";
        }
        part connectingRod : ConnectingRod {
            attribute rodId = 1003;
            attribute length = 0.145;
            attribute mass = 0.55;
            attribute material = "Forged Steel";
        }
    }
    part assembly4 : CylinderAssembly {
        attribute assemblyId = 704;
        part cylinder : Cylinder {
            attribute cylinderId = 804;
            attribute bore = 0.086;
            attribute stroke = 0.086;
            attribute displacement = 0.5;
            attribute compressionRatio = 10.5;
        }
        part piston : Piston {
            attribute pistonId = 904;
            attribute mass = 0.35;
            attribute diameter = 0.086;
            attribute material = "Aluminum Alloy";
        }
        part connectingRod : ConnectingRod {
            attribute rodId = 1004;
            attribute length = 0.145;
            attribute mass = 0.55;
            attribute material = "Forged Steel";
        }
    }
}

part sixCylEngineInstance : SixCylinderEngine {
    attribute engineId = 502;
    attribute engineType = EngineType::SixCylinder;
    attribute numberOfCylinders = 6;
    attribute totalDisplacement = 3.0;
    attribute maxPower = 220.0;
    attribute maxTorque = 380.0;
    attribute engineMass = 195.0;
    attribute thermalEfficiency = 0.36;
    attribute reliabilityScore = 0.90;
    attribute manufacturingCost = 7200.0;
    attribute fuelConsumption = 9.8;
    part crankshaft : Crankshaft {
        attribute crankshaftId = 602;
        attribute length = 0.60;
        attribute mass = 18.0;
        attribute material = "Forged Steel";
        attribute numberOfJournals = 7;
    }
    part assembly1 : CylinderAssembly {
        attribute assemblyId = 711;
        part cylinder : Cylinder {
            attribute cylinderId = 811;
            attribute bore = 0.089;
            attribute stroke = 0.080;
            attribute displacement = 0.5;
            attribute compressionRatio = 10.2;
        }
        part piston : Piston {
            attribute pistonId = 911;
            attribute mass = 0.38;
            attribute diameter = 0.089;
            attribute material = "Aluminum Alloy";
        }
        part connectingRod : ConnectingRod {
            attribute rodId = 1011;
            attribute length = 0.150;
            attribute mass = 0.58;
            attribute material = "Forged Steel";
        }
    }
    part assembly2 : CylinderAssembly {
        attribute assemblyId = 712;
        part cylinder : Cylinder {
            attribute cylinderId = 812;
            attribute bore = 0.089;
            attribute stroke = 0.080;
            attribute displacement = 0.5;
            attribute compressionRatio = 10.2;
        }
        part piston : Piston {
            attribute pistonId = 912;
            attribute mass = 0.38;
            attribute diameter = 0.089;
            attribute material = "Aluminum Alloy";
        }
        part connectingRod : ConnectingRod {
            attribute rodId = 1012;
            attribute length = 0.150;
            attribute mass = 0.58;
            attribute material = "Forged Steel";
        }
    }
    part assembly3 : CylinderAssembly {
        attribute assemblyId = 713;
        part cylinder : Cylinder {
            attribute cylinderId = 813;
            attribute bore = 0.089;
            attribute stroke = 0.080;
            attribute displacement = 0.5;
            attribute compressionRatio = 10.2;
        }
        part piston : Piston {
            attribute pistonId = 913;
            attribute mass = 0.38;
            attribute diameter = 0.089;
            attribute material = "Aluminum Alloy";
        }
        part connectingRod : ConnectingRod {
            attribute rodId = 1013;
            attribute length = 0.150;
            attribute mass = 0.58;
            attribute material = "Forged Steel";
        }
    }
    part assembly4 : CylinderAssembly {
        attribute assemblyId = 714;
        part cylinder : Cylinder {
            attribute cylinderId = 814;
            attribute bore = 0.089;
            attribute stroke = 0.080;
            attribute displacement = 0.5;
            attribute compressionRatio = 10.2;
        }
        part piston : Piston {
            attribute pistonId = 914;
            attribute mass = 0.38;
            attribute diameter = 0.089;
            attribute material = "Aluminum Alloy";
        }
        part connectingRod : ConnectingRod {
            attribute rodId = 1014;
            attribute length = 0.150;
            attribute mass = 0.58;
            attribute material = "Forged Steel";
        }
    }
    part assembly5 : CylinderAssembly {
        attribute assemblyId = 715;
        part cylinder : Cylinder {
            attribute cylinderId = 815;
            attribute bore = 0.089;
            attribute stroke = 0.080;
            attribute displacement = 0.5;
            attribute compressionRatio = 10.2;
        }
        part piston : Piston {
            attribute pistonId = 915;
            attribute mass = 0.38;
            attribute diameter = 0.089;
            attribute material = "Aluminum Alloy";
        }
        part connectingRod : ConnectingRod {
            attribute rodId = 1015;
            attribute length = 0.150;
            attribute mass = 0.58;
            attribute material = "Forged Steel";
        }
    }
    part assembly6 : CylinderAssembly {
        attribute assemblyId = 716;
        part cylinder : Cylinder {
            attribute cylinderId = 816;
            attribute bore = 0.089;
            attribute stroke = 0.080;
            attribute displacement = 0.5;
            attribute compressionRatio = 10.2;
        }
        part piston : Piston {
            attribute pistonId = 916;
            attribute mass = 0.38;
            attribute diameter = 0.089;
            attribute material = "Aluminum Alloy";
        }
        part connectingRod : ConnectingRod {
            attribute rodId = 1016;
            attribute length = 0.150;
            attribute mass = 0.58;
            attribute material = "Forged Steel";
        }
    }
}

part eightCylEngineInstance : EightCylinderEngine {
    attribute engineId = 503;
    attribute engineType = EngineType::EightCylinder;
    attribute numberOfCylinders = 8;
    attribute totalDisplacement = 5.0;
    attribute maxPower = 380.0;
    attribute maxTorque = 620.0;
    attribute engineMass = 280.0;
    attribute thermalEfficiency = 0.30;
    attribute reliabilityScore = 0.85;
    attribute manufacturingCost = 12000.0;
    attribute fuelConsumption = 14.5;
    part crankshaft : Crankshaft {
        attribute crankshaftId = 603;
        attribute length = 0.75;
        attribute mass = 26.0;
        attribute material = "Forged Steel";
        attribute numberOfJournals = 9;
    }
    part assembly1 : CylinderAssembly {
        attribute assemblyId = 721;
        part cylinder : Cylinder {
            attribute cylinderId = 821;
            attribute bore = 0.099;
            attribute stroke = 0.082;
            attribute displacement = 0.625;
            attribute compressionRatio = 9.8;
        }
        part piston : Piston {
            attribute pistonId = 921;
            attribute mass = 0.45;
            attribute diameter = 0.099;
            attribute material = "Forged Aluminum";
        }
        part connectingRod : ConnectingRod {
            attribute rodId = 1021;
            attribute length = 0.158;
            attribute mass = 0.68;
            attribute material = "Forged Steel";
        }
    }
    part assembly2 : CylinderAssembly {
        attribute assemblyId = 722;
        part cylinder : Cylinder {
            attribute cylinderId = 822;
            attribute bore = 0.099;
            attribute stroke = 0.082;
            attribute displacement = 0.625;
            attribute compressionRatio = 9.8;
        }
        part piston : Piston {
            attribute pistonId = 922;
            attribute mass = 0.45;
            attribute diameter = 0.099;
            attribute material = "Forged Aluminum";
        }
        part connectingRod : ConnectingRod {
            attribute rodId = 1022;
            attribute length = 0.158;
            attribute mass = 0.68;
            attribute material = "Forged Steel";
        }
    }
    part assembly3 : CylinderAssembly {
        attribute assemblyId = 723;
        part cylinder : Cylinder {
            attribute cylinderId = 823;
            attribute bore = 0.099;
            attribute stroke = 0.082;
            attribute displacement = 0.625;
            attribute compressionRatio = 9.8;
        }
        part piston : Piston {
            attribute pistonId = 923;
            attribute mass = 0.45;
            attribute diameter = 0.099;
            attribute material = "Forged Aluminum";
        }
        part connectingRod : ConnectingRod {
            attribute rodId = 1023;
            attribute length = 0.158;
            attribute mass = 0.68;
            attribute material = "Forged Steel";
        }
    }
    part assembly4 : CylinderAssembly {
        attribute assemblyId = 724;
        part cylinder : Cylinder {
            attribute cylinderId = 824;
            attribute bore = 0.099;
            attribute stroke = 0.082;
            attribute displacement = 0.625;
            attribute compressionRatio = 9.8;
        }
        part piston : Piston {
            attribute pistonId = 924;
            attribute mass = 0.45;
            attribute diameter = 0.099;
            attribute material = "Forged Aluminum";
        }
        part connectingRod : ConnectingRod {
            attribute rodId = 1024;
            attribute length = 0.158;
            attribute mass = 0.68;
            attribute material = "Forged Steel";
        }
    }
    part assembly5 : CylinderAssembly {
        attribute assemblyId = 725;
        part cylinder : Cylinder {
            attribute cylinderId = 825;
            attribute bore = 0.099;
            attribute stroke = 0.082;
            attribute displacement = 0.625;
            attribute compressionRatio = 9.8;
        }
        part piston : Piston {
            attribute pistonId = 925;
            attribute mass = 0.45;
            attribute diameter = 0.099;
            attribute material = "Forged Aluminum";
        }
        part connectingRod : ConnectingRod {
            attribute rodId = 1025;
            attribute length = 0.158;
            attribute mass = 0.68;
            attribute material = "Forged Steel";
        }
    }
    part assembly6 : CylinderAssembly {
        attribute assemblyId = 726;
        part cylinder : Cylinder {
            attribute cylinderId = 826;
            attribute bore = 0.099;
            attribute stroke = 0.082;
            attribute displacement = 0.625;
            attribute compressionRatio = 9.8;
        }
        part piston : Piston {
            attribute pistonId = 926;
            attribute mass = 0.45;
            attribute diameter = 0.099;
            attribute material = "Forged Aluminum";
        }
        part connectingRod : ConnectingRod {
            attribute rodId = 1026;
            attribute length = 0.158;
            attribute mass = 0.68;
            attribute material = "Forged Steel";
        }
    }
    part assembly7 : CylinderAssembly {
        attribute assemblyId = 727;
        part cylinder : Cylinder {
            attribute cylinderId = 827;
            attribute bore = 0.099;
            attribute stroke = 0.082;
            attribute displacement = 0.625;
            attribute compressionRatio = 9.8;
        }
        part piston : Piston {
            attribute pistonId = 927;
            attribute mass = 0.45;
            attribute diameter = 0.099;
            attribute material = "Forged Aluminum";
        }
        part connectingRod : ConnectingRod {
            attribute rodId = 1027;
            attribute length = 0.158;
            attribute mass = 0.68;
            attribute material = "Forged Steel";
        }
    }
    part assembly8 : CylinderAssembly {
        attribute assemblyId = 728;
        part cylinder : Cylinder {
            attribute cylinderId = 828;
            attribute bore = 0.099;
            attribute stroke = 0.082;
            attribute displacement = 0.625;
            attribute compressionRatio = 9.8;
        }
        part piston : Piston {
            attribute pistonId = 928;
            attribute mass = 0.45;
            attribute diameter = 0.099;
            attribute material = "Forged Aluminum";
        }
        part connectingRod : ConnectingRod {
            attribute rodId = 1028;
            attribute length = 0.158;
            attribute mass = 0.68;
            attribute material = "Forged Steel";
        }
    }
}

part systemTradeoffAnalysis : TradeoffAnalysis {
    attribute analysisId = 501;
    attribute analysisName = "System Level Engine Tradeoff Analysis";
    attribute numberOfCandidates = 3;
    attribute bestCandidateId = 1;
    attribute bestCompositeScore = 0.87;
    attribute recommendation = RecommendationStatus::Recommended;
    attribute selectionRationale = "Four-cylinder 2.0L engine selected as