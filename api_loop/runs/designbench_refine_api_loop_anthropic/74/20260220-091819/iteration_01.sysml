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
        attribute redefines power : Real;
        attribute redefines mass : Real;
        attribute redefines efficiency : Real;
        attribute redefines cost : Real;
        attribute redefines comprehensiveScore : Real;
    }

    part def SixCylinderEngine :> Engine {
        attribute redefines engineType = EngineType::SixCylinder;
        attribute redefines power : Real;
        attribute redefines mass : Real;
        attribute redefines efficiency : Real;
        attribute redefines cost : Real;
        attribute redefines comprehensiveScore : Real;
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
            attribute weightPower : Real = 0.35;
            attribute weightMass : Real = 0.20;
            attribute weightEfficiency : Real = 0.30;
            attribute weightCost : Real = 0.15;
        }

        part fourCylinderCandidate : FourCylinderEngine {
            attribute redefines power : Real = 110.0;
            attribute redefines mass : Real = 140.0;
            attribute redefines efficiency : Real = 0.38;
            attribute redefines cost : Real = 8000.0;
            attribute redefines comprehensiveScore : Real = 0.72;
        }

        part sixCylinderCandidate : SixCylinderEngine {
            attribute redefines power : Real = 180.0;
            attribute redefines mass : Real = 210.0;
            attribute redefines efficiency : Real = 0.42;
            attribute redefines cost : Real = 14000.0;
            attribute redefines comprehensiveScore : Real = 0.81;
        }

        part selectionResult : SelectionResult {
            attribute selectedEngineType : EngineType = EngineType::SixCylinder;
            attribute selectedPower : Real = 180.0;
            attribute selectedMass : Real = 210.0;
            attribute selectedEfficiency : Real = 0.42;
            attribute selectedCost : Real = 14000.0;
            attribute selectedScore : Real = 0.81;
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

    part def EngineSelectionSystem {
        attribute systemName : String = "EngineSelectionSystem";

        part analysis : EngineSelectionAnalysis {
            attribute analysisName : String = "ComprehensiveEngineEvaluation";
        }

        part dataSource : EngineDataSource;
        part dataConsumer : EngineDataConsumer;

        connect dataSource.outputPort to dataConsumer.inputPort;
    }
}