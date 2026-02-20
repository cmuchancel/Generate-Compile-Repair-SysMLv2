package VehicleWeightValidation {
    private import ScalarValues::*;

    enum def EvaluationResult {
        enum Qualified;
        enum Unqualified;
    }

    part def Vehicle {
        attribute curbWeight : Real;
    }

    part def WeightDataCollector {
        port collectDataPort;
    }

    part def WeightDataProcessor {
        port processDataPort;
    }

    part def WeightEvaluator {
        attribute maxAllowedWeight : Real = 2500.0;
        port evaluatePort;
    }

    part def WeightValidationSystem {
        part vehicle : Vehicle;
        part collector : WeightDataCollector;
        part processor : WeightDataProcessor;
        part evaluator : WeightEvaluator;
        attribute result : EvaluationResult;

        connect collector.collectDataPort to processor.processDataPort;
        connect processor.processDataPort to evaluator.evaluatePort;
    }

    item def WeightData {
        attribute rawWeight : Real;
        attribute processedWeight : Real;
    }
}