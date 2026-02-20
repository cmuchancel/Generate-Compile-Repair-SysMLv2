package RiskMetadataModel {
    private import ScalarValues::*;

    enum def SeverityLevel {
        Low;
        Medium;
        High;
        Critical;
    }

    enum def FailureCategory {
        Functional;
        Safety;
        Security;
        Performance;
        Reliability;
    }

    part def Cause {
        attribute name : String;
        attribute description : String;
        attribute probability : Real;
    }

    part def Failure {
        attribute name : String;
        attribute description : String;
        attribute category : FailureCategory;
        attribute severity : SeverityLevel;
    }

    part def Scenario {
        attribute name : String;
        attribute description : String;
        part causes : Cause[0..*];
        part failures : Failure[0..*];
    }

    part def ScenarioCausalLink {
        part source : Scenario;
        part target : Scenario;
        attribute description : String;
        attribute propagationProbability : Real;
    }

    part def RiskScenario {
        attribute name : String;
        attribute description : String;
        part scenarios : Scenario[1..*];
        part causes : Cause[0..*];
        part failures : Failure[0..*];
        part links : ScenarioCausalLink[0..*];
    }

    part def RiskMetadataRepository {
        attribute metadataFormat : String;
        part scenarios : Scenario[0..*];
        part causes : Cause[0..*];
        part failures : Failure[0..*];
        part links : ScenarioCausalLink[0..*];
        part riskScenarios : RiskScenario[0..*];
    }

    part repository : RiskMetadataRepository;
}