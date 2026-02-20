package 'EngineSelectionAnalysis' {
	private import ScalarValues::*;
	private import NumericalFunctions::*;
	private import ISQ::*;
	private import SI::kg;
	calc def EngineEvaluationScore {
		in normalizedPower : Real;
		in normalizedEfficiency : Real;
		in normalizedMass : Real;
		in normalizedCost : Real;
		return : Real = (normalizedPower * normalizedEfficiency) - (normalizedMass + normalizedCost);
	}
	part def EngineCandidate {
		attribute name : String;
		attribute cylinderCount : Integer;
		attribute ratedPower : PowerValue;
		attribute mass : MassValue;
		attribute efficiency : Real;
		attribute cost : Real;
		attribute normalizedPower : Real;
		attribute normalizedEfficiency : Real;
		attribute normalizedMass : Real;
		attribute normalizedCost : Real;
		attribute evaluationScore : Real default EngineEvaluationScore(
			normalizedPower,
			normalizedEfficiency,
			normalizedMass,
			normalizedCost
		);
	}
	part def EngineTypeGroup {
		attribute cylinderCount : Integer;
		part candidates : EngineCandidate[*];
		attribute aggregatedPower : PowerValue default sum(candidates.ratedPower);
		attribute aggregatedMass : MassValue default sum(candidates.mass);
		attribute aggregatedEfficiency : Real;
		attribute aggregatedCost : Real default sum(candidates.cost);
		attribute bestScore : Real;
	}
	part def EngineSelectionSystem {
		part fourCylinderGroup : EngineTypeGroup {
			attribute :>> cylinderCount default 4;
		}
		part sixCylinderGroup : EngineTypeGroup {
			attribute :>> cylinderCount default 6;
		}
		attribute evaluationFunctionDescription : String;
		ref part recommendedEngine : EngineCandidate;
		attribute recommendedEngineScore : Real default recommendedEngine.evaluationScore;
		attribute selectionMaximizesPerformanceAndEconomics : Boolean default
			recommendedEngineScore >= fourCylinderGroup.bestScore and
			recommendedEngineScore >= sixCylinderGroup.bestScore;
	}
}
