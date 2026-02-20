package RiskAndFailureManagement {
    private import ScalarValues::*;

    enum def ProbabilityLevel {
        enum VERY_LOW
        enum LOW
        enum MEDIUM
        enum HIGH
        enum VERY_HIGH
    }

    enum def SeverityLevel {
        enum NEGLIGIBLE
        enum MINOR
        enum MODERATE
        enum MAJOR
        enum CATASTROPHIC
    }

    part def Situation {
        attribute description : String;
        attribute causes : Cause[0..*];
    }

    part def Cause {
        attribute description : String;
        attribute probability : ProbabilityLevel;
        attribute resultingFailures : Failure[0..*];
    }

    part def Failure {
        attribute description : String;
        attribute severity : SeverityLevel;
    }

    part def RiskScenario {
        attribute description : String;
        attribute situations : Situation[0..*];
        attribute causes : Cause[0..*];
        attribute failures : Failure[0..*];
    }

    part system : RiskAndFailureManagement::RiskScenario {
        part situation1 : Situation;
        part cause1 : Cause;
        part failure1 : Failure;

        connect situation1.causes to cause1;
        connect cause1.resultingFailures to failure1;
    }
}