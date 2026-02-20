package RiskManagement {
    private import ScalarValues::*;

    enum def SeverityLevel {
        enum Low;
        enum Medium;
        enum High;
        enum Critical;
    }

    enum def ProbabilityLevel {
        enum VeryLow;
        enum Low;
        enum Medium;
        enum High;
        enum VeryHigh;
    }

    part def Situation {
        attribute description : String;
        port in causePort : Cause[0..*];
        port out effectPort : Situation[0..*];
        port out failurePort : Failure[0..*];
    }

    part def Cause {
        attribute description : String;
        attribute probability : ProbabilityLevel;
        port out leadsTo : Situation[0..*];
    }

    part def Failure {
        attribute description : String;
        attribute severity : SeverityLevel;
        port in causedBy : Situation[0..*];
    }

    part def RiskScenario {
        part rootCause : Cause;
        part intermediateSituation : Situation;
        part finalFailure : Failure;

        connect rootCause.leadsTo to intermediateSituation.causePort;
        connect intermediateSituation.failurePort to finalFailure.causedBy;
    }
}