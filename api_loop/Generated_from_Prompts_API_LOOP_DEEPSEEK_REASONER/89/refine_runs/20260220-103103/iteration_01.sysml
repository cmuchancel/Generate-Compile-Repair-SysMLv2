package RiskManagement {
    private import ScalarValues::*;

    enum def SeverityLevel {
        enum Low,
        enum Medium,
        enum High,
        enum Critical
    }

    enum def ProbabilityLevel {
        enum VeryLow,
        enum Low,
        enum Medium,
        enum High,
        enum VeryHigh
    }

    part def Situation {
        attribute description : String;
        port input causePort : Cause[0..*];
        port output effectPort : Situation[0..*];
        port output failurePort : Failure[0..*];
    }

    part def Cause {
        attribute description : String;
        attribute probability : ProbabilityLevel;
        port output leadsTo : Situation[0..*];
    }

    part def Failure {
        attribute description : String;
        attribute severity : SeverityLevel;
        port input causedBy : Situation[0..*];
    }

    part def RiskScenario {
        part rootCause : Cause;
        part intermediateSituation : Situation;
        part finalFailure : Failure;

        connect rootCause.leadsTo to intermediateSituation.causePort;
        connect intermediateSituation.effectPort to intermediateSituation.causePort;
        connect intermediateSituation.failurePort to finalFailure.causedBy;
    }
}