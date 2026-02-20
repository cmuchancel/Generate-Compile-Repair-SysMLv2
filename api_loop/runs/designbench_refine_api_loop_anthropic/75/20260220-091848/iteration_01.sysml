package VehicleWeightCheck {
    private import ScalarValues::*;

    enum def EvaluationResult {
        enum Qualified;
        enum Unqualified;
    }

    item def WeightData {
        attribute rawWeight : Real;
    }

    item def ProcessedData {
        attribute processedWeight : Real;
    }

    part def DataCollectionStep {
        attribute collectedWeight : Real;
        port dataOut : DataOutPort;
    }

    part def DataProcessingStep {
        attribute processedWeight : Real;
        port dataIn : DataInPort;
        port dataOut : DataOutPort;
    }

    part def EvaluationStep {
        attribute maxCurbWeight : Real = 2500.0;
        attribute evaluatedWeight : Real;
        attribute result : EvaluationResult;
        port dataIn : DataInPort;
    }

    port def DataOutPort {
        attribute value : Real;
    }

    port def DataInPort {
        attribute value : Real;
    }

    part def Vehicle {
        attribute curbWeight : Real;
        attribute maxAllowedWeight : Real = 2500.0;
    }

    part def WeightCheckProcess {
        part collection : DataCollectionStep;
        part processing : DataProcessingStep;
        part evaluation : EvaluationStep;

        connect collection.dataOut to processing.dataIn;
        connect processing.dataOut to evaluation.dataIn;
    }

    part vehicleWeightCheckSystem {
        part vehicle : Vehicle {
            attribute :>> curbWeight = 2200.0;
        }
        part checkProcess : WeightCheckProcess;
    }
}