package EngineSelectionSystem {
    private import ScalarValues::*;

    enum def EngineType {
        enum FourCylinder;
        enum SixCylinder;
    }

    part def Engine {
        attribute engineType : EngineType;
        attribute power : Real;
        attribute mass : Real;
        attribute efficiency : Real;
        attribute cost : Real;
        attribute comprehensiveScore : Real;
    }

    part def FourCylinderEngine :> Engine {
        attribute redefines engineType = EngineType::FourCylinder;
    }

    part def SixCylinderEngine :> Engine {
        attribute redefines engineType = EngineType::SixCylinder;
    }

    part def EvaluationFunction {
        attribute weightPower : Real;
        attribute weightMass : Real;
        attribute weightEfficiency : Real;
        attribute weightCost : Real;
    }

    part def SelectionResult {
        attribute selectedEngineType : EngineType;
        attribute selectedPower : Real;
        attribute selectedMass : Real;
        attribute selectedEfficiency : Real;
        attribute selectedCost : Real;
        attribute selectedScore : Real;
    }

    part def EngineSelectionAnalysis {
        attribute analysisName : String;

        part evaluationFunction : EvaluationFunction {
            attribute redefines weightPower = 0.35;
            attribute redefines weightMass = 0.20;
            attribute redefines weightEfficiency = 0.30;
            attribute redefines weightCost = 0.15;
        }

        part fourCylinderCandidate : FourCylinderEngine {
            attribute redefines power = 110.0;
            attribute redefines mass = 140.0;
            attribute redefines efficiency = 0.38;
            attribute redefines cost = 8000.0;
            attribute redefines comprehensiveScore = 0.72;
        }

        part sixCylinderCandidate : SixCylinderEngine {
            attribute redefines power = 180.0;
            attribute redefines mass = 210.0;
            attribute redefines efficiency = 0.42;
            attribute redefines cost = 14000.0;
            attribute redefines comprehensiveScore = 0.81;
        }

        part selectionResult : SelectionResult {
            attribute redefines selectedEngineType = EngineType::SixCylinder;
            attribute redefines selectedPower = 180.0;
            attribute redefines selectedMass = 210.0;
            attribute redefines selectedEfficiency = 0.42;
            attribute redefines selectedCost = 14000.0;
            attribute redefines selectedScore = 0.81;
        }
    }

    port def EngineDataPort {
        attribute portPower : Real;
        attribute portMass : Real;
        attribute portEfficiency : Real;
        attribute portCost : Real;
        attribute portScore : Real;
    }

    part def EngineDataSource {
        port outputPort : EngineDataPort;
    }

    part def EngineDataConsumer {
        port inputPort : EngineDataPort;
    }

    part def EngineSelectionSystemDef {
        attribute systemName : String = "EngineSelectionSystem";

        attribute analysisName : String = "ComprehensiveEngineEvaluation";

        part evaluationFunction : EvaluationFunction {
            attribute redefines weightPower = 0.35;
            attribute redefines weightMass = 0.20;
            attribute redefines weightEfficiency = 0.30;
            attribute redefines weightCost = 0.15;
        }

        part fourCylinderCandidate : FourCylinderEngine {
            attribute redefines power = 110.0;
            attribute redefines mass = 140.0;
            attribute redefines efficiency = 0.38;
            attribute redefines cost = 8000.0;
            attribute redefines comprehensiveScore = 0.72;
        }

        part sixCylinderCandidate : SixCylinderEngine {
            attribute redefines power = 180.0;
            attribute redefines mass = 210.0;
            attribute redefines efficiency = 0.42;
            attribute redefines cost = 14000.0;
            attribute redefines comprehensiveScore = 0.81;
        }

        part selectionResult : SelectionResult {
            attribute redefines selectedEngineType = EngineType::SixCylinder;
            attribute redefines selectedPower = 180.0;
            attribute redefines selectedMass = 210.0;
            attribute redefines selectedEfficiency = 0.42;
            attribute redefines selectedCost = 14000.0;
            attribute redefines selectedScore = 0.81;
        }

        part dataSource : EngineDataSource;
        part dataConsumer : EngineDataConsumer;

        connect dataSource.outputPort to dataConsumer.inputPort;
    }
}