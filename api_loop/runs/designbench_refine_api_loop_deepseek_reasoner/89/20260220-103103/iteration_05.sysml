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
        port causePort : Cause[0..*];
        port effectPort : Situation[0..*];
        port failurePort : Failure[0..*];
    }

    part def Cause {
        attribute description : String;
        attribute probability : ProbabilityLevel;
        port leadsTo : Situation[0..*];
    }

    part def Failure {
        attribute description : String;
        attribute severity : SeverityLevel;
        port causedBy : Situation[0..*];
    }

    part def RiskScenario {
        part rootCause : Cause;
        part intermediateSituation : Situation;
        part finalFailure : Failure;

        connect rootCause.leadsTo to intermediateSituation.causePort;
        connect intermediateSituation.failurePort to finalFailure.causedBy;
    }
}