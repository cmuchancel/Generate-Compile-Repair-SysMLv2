package 'RiskMetadataModel' {
	private import ScalarValues::*;
	part def Cause {
		attribute name : String;
		attribute rationale : String;
	}
	part def Failure {
		attribute name : String;
		attribute description : String;
	}
	part def Scenario {
		attribute summary : String;
		part causes : CauseAssessment[*];
		part failures : FailureImpact[*];
	}
	part def CauseAssessment {
		ref part cause : Cause;
		attribute probability : Real;
	}
	part def FailureImpact {
		ref part failure : Failure;
		attribute severityLevel : Integer;
	}
	connection def ScenarioCausation {
		end source : Scenario[1];
		end consequence : Scenario[1];
	}
	part def CompositeRiskScenario {
		part scenarios : Scenario[1..*];
		part contributingCauses : CauseAssessment[*];
		part resultingFailures : FailureImpact[*];
		part propagationChain : ScenarioCausation[*];
	}
}
