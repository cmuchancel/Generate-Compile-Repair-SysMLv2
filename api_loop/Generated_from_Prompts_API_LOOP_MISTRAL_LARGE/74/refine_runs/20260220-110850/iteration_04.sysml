package EngineSelectionAnalysis {
    private import ScalarValues::*;

    enum def EngineType {
        enum FourCylinder;
        enum SixCylinder;
    }

    part def Engine {
        attribute power : Real;
        attribute mass : Real;
        attribute efficiency : Real;
        attribute cost : Real;
        attribute engineType : EngineType;
    }

    part def EvaluationFunction {
        attribute weightPower : Real = 0.3;
        attribute weightMass : Real = 0.2;
        attribute weightEfficiency : Real = 0.3;
        attribute weightCost : Real = 0.2;

        attribute evaluate : Real;
    }

    part def EngineSelectionSystem {
        part candidateEngines : Engine[0..*];
        part evaluator : EvaluationFunction;
        attribute selectedEngine : Engine;

        port def SelectionOutput {
            attribute recommendedEngine : Engine;
        }

        port output : SelectionOutput;

        attribute selectOptimalEngine : Engine;
    }
}