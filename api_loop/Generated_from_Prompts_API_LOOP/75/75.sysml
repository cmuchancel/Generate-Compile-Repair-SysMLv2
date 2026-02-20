package VehicleQualityModel {
    private import ScalarValues::*;

    enum def EvaluationResultEnum {
        enum Qualified;
        enum Unqualified;
    }

    item def WeightData {
        attribute value : Real;
    }

    item def ProcessedWeightData {
        attribute value : Real;
    }

    part def Vehicle {
        attribute curbWeight : Real;
    }

    part def DataCollection {
        port outWeight : WeightData;
    }

    part def DataProcessing {
        port inWeight : WeightData;
        port outProcessed : ProcessedWeightData;
    }

    part def Evaluation {
        port inProcessed : ProcessedWeightData;
        attribute maxCurbWeight : Real = 2500;
        attribute result : EvaluationResultEnum;
    }

    part def QualityCheckSystem {
        part vehicle : Vehicle;
        part collector : DataCollection;
        part processor : DataProcessing;
        part evaluator : Evaluation;

        connect collector.outWeight to processor.inWeight;
        connect processor.outProcessed to evaluator.inProcessed;
    }
}