package 'VehicleCurbWeightVerification' {
	private import ISQ::*;
	private import ScalarValues::*;
	private import SI::kg;
	part def VehicleWeightCollection {
		attribute collectedCurbWeight : MassValue;
		attribute collectionTimestamp : TimeValue;
		attribute collectionMethod : String;
	}
	part def WeightDataProcessing {
		attribute inputCurbWeight : MassValue;
		attribute processedCurbWeight : MassValue;
		attribute processingApproach : String;
	}
	part def WeightEvaluation {
		attribute maximumCurbWeight : MassValue = 2500[kg];
		attribute evaluatedCurbWeight : MassValue;
		attribute meetsRequirement : Boolean default evaluatedCurbWeight <= maximumCurbWeight;
		attribute evaluationResult : String;
	}
	part def VehicleWeightTestWorkflow {
		part collection : VehicleWeightCollection;
		part processing : WeightDataProcessing {
			attribute :>> inputCurbWeight default collection.collectedCurbWeight;
			attribute :>> processedCurbWeight default collection.collectedCurbWeight;
		}
		part evaluation : WeightEvaluation {
			attribute :>> evaluatedCurbWeight default processing.processedCurbWeight;
		}
		attribute qualified : Boolean default evaluation.meetsRequirement;
		attribute evaluationSummary : String;
	}
}
