package RiskAndFailureManagement {
    private import ScalarValues::*;

    enum def SeverityLevel {
        enum LOW;
        enum MEDIUM;
        enum HIGH;
        enum CRITICAL;
    }

    enum def ProbabilityLevel {
        enum VERY_LOW;
        enum LOW;
        enum MEDIUM;
        enum HIGH;
        enum VERY_HIGH;
    }

    enum def RiskCategory {
        enum OPERATIONAL;
        enum TECHNICAL;
        enum ENVIRONMENTAL;
        enum HUMAN;
        enum ORGANIZATIONAL;
    }

    part def Situation {
        attribute situationId : String;
        attribute situationName : String;
        attribute description : String;
        attribute category : RiskCategory;
    }

    part def Cause {
        attribute causeId : String;
        attribute causeName : String;
        attribute description : String;
        attribute probability : ProbabilityLevel;
        attribute probabilityValue : Real;
    }

    part def Failure {
        attribute failureId : String;
        attribute failureName : String;
        attribute description : String;
        attribute severity : SeverityLevel;
        attribute impactDescription : String;
    }

    part def CausalRelationship {
        attribute relationshipId : String;
        attribute relationshipDescription : String;
        attribute sequenceOrder : Integer;
        attribute impactMechanism : String;
    }

    part def RiskScenario {
        attribute scenarioId : String;
        attribute scenarioName : String;
        attribute description : String;
        attribute overallRiskLevel : SeverityLevel;
        attribute evaluationResult : String;
        attribute mitigationStrategy : String;

        part scenarioSituation : Situation;
        part scenarioCauses : Cause[1..*];
        part scenarioFailures : Failure[1..*];
        part causalLinks : CausalRelationship[0..*];
    }

    part def RiskControl {
        attribute controlId : String;
        attribute controlName : String;
        attribute controlDescription : String;
        attribute effectiveness : SeverityLevel;
        attribute implementationStatus : String;
    }

    part def DecisionSupport {
        attribute decisionId : String;
        attribute decisionName : String;
        attribute decisionBasis : String;
        attribute recommendedAction : String;
        attribute priority : SeverityLevel;
    }

    port def RiskDataPort {
        attribute dataType : String;
        attribute dataVersion : String;
    }

    part def RiskAndFailureSystem {
        attribute systemId : String;
        attribute systemName : String;
        attribute systemVersion : String;
        attribute systemDescription : String;

        port riskDataInput : RiskDataPort;
        port riskDataOutput : RiskDataPort;

        part situations : Situation[0..*];
        part causes : Cause[0..*];
        part failures : Failure[0..*];
        part causalRelationships : CausalRelationship[0..*];
        part riskScenarios : RiskScenario[0..*];
        part riskControls : RiskControl[0..*];
        part decisionSupports : DecisionSupport[0..*];
    }

    part def RiskEvaluator {
        attribute evaluatorId : String;
        attribute evaluatorName : String;
        attribute evaluationMethod : String;

        port evaluationInput : RiskDataPort;
        port evaluationOutput : RiskDataPort;
    }

    part def RiskManager {
        attribute managerId : String;
        attribute managerName : String;

        port managerInput : RiskDataPort;
        port managerOutput : RiskDataPort;

        part managedSystem : RiskAndFailureSystem;
        part evaluator : RiskEvaluator;

        connect managedSystem.riskDataOutput to evaluator.evaluationInput;
        connect evaluator.evaluationOutput to managedSystem.riskDataInput;
    }
}