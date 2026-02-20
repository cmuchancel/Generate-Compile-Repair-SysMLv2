package VehiclePowertrain {
    private import ScalarValues::*;

    enum def EngineType {
        FourCylinder,
        SixCylinder,
        EightCylinder,
        Electric
    }

    part def EngineComponent;

    part def Cylinder : EngineComponent {
    }

    part def Piston : EngineComponent {
    }

    part def ConnectingRod : EngineComponent {
    }

    part def Crankshaft : EngineComponent {
    }

    part def Engine {
        attribute engineType : EngineType;
        attribute power : Real;
        attribute mass : Real;
        attribute efficiency : Real;
        attribute reliability : Real;
        attribute cost : Real;

        part cylinders : Cylinder[*];
        part pistons : Piston[*];
        part connectingRods : ConnectingRod[*];
        part crankshaft : Crankshaft[0..1];
    }

    part def EvaluationModel {
        attribute weightingPower : Real = 0.25;
        attribute weightingMass : Real = 0.15;
        attribute weightingEfficiency : Real = 0.25;
        attribute weightingReliability : Real = 0.20;
        attribute weightingCost : Real = 0.15;

        part engineOptions : Engine[*];
        attribute selectedEngine : Engine;
    }

    part def PowertrainSystem {
        part engine : Engine;
        part evaluator : EvaluationModel;

        connect engine to evaluator.selectedEngine;
    }

    item def KPI {
        attribute powerScore : Real;
        attribute massScore : Real;
        attribute efficiencyScore : Real;
        attribute reliabilityScore : Real;
        attribute costScore : Real;
        attribute overallScore : Real;
    }

    part def TradeOffAnalysis {
        part evaluationModel : EvaluationModel;
        attribute kpiResults : KPI[*];
    }
}