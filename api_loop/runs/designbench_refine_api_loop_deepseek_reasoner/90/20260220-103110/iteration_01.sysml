package RiskMetadataModel {
    private import ScalarValues::*;

    enum def ProbabilityLevel {
        Rare,
        Unlikely,
        Possible,
        Likely,
        Frequent
    }

    enum def SeverityLevel {
        Negligible,
        Minor,
        Moderate,
        Critical,
        Catastrophic
    }

    part def Cause {
        attribute id : String;
        attribute description : String;
        attribute probability : ProbabilityLevel = ProbabilityLevel::Possible;
    }

    part def Failure {
        attribute id : String;
        attribute description : String;
        attribute severity : SeverityLevel = SeverityLevel::Moderate;
    }

    part def Scenario {
        attribute id : String;
        attribute description : String;
        part causes : Cause[*];
        part failures : Failure[*];
        port input : ScenarioPort[0..*];
        port output : ScenarioPort[0..*];
    }

    port def ScenarioPort {
        attribute direction : String;
    }

    part def RiskScenario {
        attribute id : String;
        attribute description : String;
        part scenarios : Scenario[*];
        part causes : Cause[*];
        part failures : Failure[*];
    }

    part riskSystem : RiskScenario[1];
}