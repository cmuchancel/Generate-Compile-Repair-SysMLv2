package VehicleWeightCheck {
    private import ScalarValues::*;

    enum def EvaluationResult {
        enum qualified;
        enum unqualified;
    }

    item def RawWeightData {
        attribute value : Real;
    }

    item def ProcessedWeightData {
        attribute value : Real;
    }

    part def Vehicle {
        attribute curbWeight : Real;
    }

    port def DataCollectPort {
        attribute rawData : RawWeightData;
    }

    port def DataProcessPort {
        attribute processedData : ProcessedWeightData;
    }

    port def EvaluationPort {
        attribute result : EvaluationResult;
    }

    part def DataCollector {
        port output : DataCollectPort;
    }

    part def DataProcessor {
        port input : DataCollectPort;
        port output : DataProcessPort;
    }

    part def Evaluator {
        port input : DataProcessPort;
        port output : EvaluationPort;
        attribute weightLimit : Real = 2500;
    }

    part def WeightCheckSystem {
        part vehicle : Vehicle;
        part collector : DataCollector;
        part processor : DataProcessor;
        part evaluator : Evaluator;

        connect vehicle to collector;
        connect collector to processor;
        connect processor to evaluator;
    }
}