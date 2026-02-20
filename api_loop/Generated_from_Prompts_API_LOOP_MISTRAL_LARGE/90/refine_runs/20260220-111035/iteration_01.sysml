package RiskMetadataModeling {
    private import ScalarValues::*;

    enum def SeverityLevel {
        enum LOW;
        enum MEDIUM;
        enum HIGH;
        enum CRITICAL;
    }

    enum def ProbabilityLevel {
        enum RARE;
        enum UNLIKELY;
        enum POSSIBLE;
        enum LIKELY;
        enum CERTAIN;
    }

    part def Scenario {
        attribute description : String;
        attribute causes : Cause[0..*];
        attribute failures : Failure[0..*];
        attribute relatedScenarios : Scenario[0..*];
    }

    part def Cause {
        attribute description : String;
        attribute probability : ProbabilityLevel;
    }

    part def Failure {
        attribute description : String;
        attribute severity : SeverityLevel;
    }

    part def RiskScenario {
        attribute description : String;
        attribute scenarios : Scenario[1..*];
        attribute causes : Cause[0..*];
        attribute failures : Failure[0..*];
    }

    part def RiskModel {
        part scenarios : Scenario[0..*];
        part causes : Cause[0..*];
        part failures : Failure[0..*];
        part riskScenarios : RiskScenario[0..*];
    }

    part riskModel : RiskModel;

    connect riskModel.scenarios to riskModel.riskScenarios.scenarios;
    connect riskModel.causes to riskModel.riskScenarios.causes;
    connect riskModel.failures to riskModel.riskScenarios.failures;
}