package RiskManagement
{
    private import ScalarValues::*;

    enum def SeverityLevel {
        Low;
        Medium;
        High;
        Critical;
    }

    enum def ProbabilityLevel {
        Rare;
        Unlikely;
        Possible;
        Likely;
        Certain;
    }

    item def Situation {
        attribute description : String;
    }

    item def Cause {
        attribute description : String;
        attribute probability : ProbabilityLevel;
    }

    item def Failure {
        attribute description : String;
        attribute severity : SeverityLevel;
    }

    part def CausalLink {
        part fromSituation : Situation;
        part cause : Cause;
        part toFailure : Failure;
    }

    part def RiskScenario {
        part situation : Situation;
        part cause : Cause;
        part failure : Failure;
        part link : CausalLink;
    }

    part def RiskModel {
        part scenario1 : RiskScenario;
        part scenario2 : RiskScenario;

        connect scenario1.situation to scenario1.link.fromSituation;
        connect scenario1.cause to scenario1.link.cause;
        connect scenario1.failure to scenario1.link.toFailure;

        connect scenario2.situation to scenario2.link.fromSituation;
        connect scenario2.cause to scenario2.link.cause;
        connect scenario2.failure to scenario2.link.toFailure;
    }
}