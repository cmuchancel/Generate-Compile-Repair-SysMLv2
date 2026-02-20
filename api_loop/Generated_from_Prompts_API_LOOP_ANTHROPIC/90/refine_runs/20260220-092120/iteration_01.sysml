package RiskMetadataModel {
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
        enum ALMOST_CERTAIN;
    }

    enum def FailureType {
        enum FUNCTIONAL;
        enum PERFORMANCE;
        enum SAFETY;
        enum SECURITY;
        enum AVAILABILITY;
    }

    part def Cause {
        attribute causeId : String;
        attribute causeDescription : String;
        attribute probability : ProbabilityLevel;
        attribute probabilityValue : Real;
    }

    part def Failure {
        attribute failureId : String;
        attribute failureDescription : String;
        attribute failureType : FailureType;
        attribute severity : SeverityLevel;
        attribute severityScore : Integer;
    }

    part def Scenario {
        attribute scenarioId : String;
        attribute scenarioName : String;
        attribute scenarioDescription : String;
        attribute likelihood : ProbabilityLevel;
        part scenarioCauses : Cause[0..*];
        part scenarioFailures : Failure[0..*];
    }

    part def CausalRelationship {
        attribute relationshipId : String;
        attribute relationshipDescription : String;
        attribute propagationStrength : Real;
        part sourceScenario : Scenario[1];
        part targetScenario : Scenario[1];
    }

    part def RiskScenario {
        attribute riskScenarioId : String;
        attribute riskScenarioName : String;
        attribute riskScenarioDescription : String;
        attribute overallSeverity : SeverityLevel;
        attribute overallProbability : ProbabilityLevel;
        attribute riskScore : Real;
        attribute analysisNotes : String;
        attribute decisionSupport : String;
        part composedScenarios : Scenario[1..*];
        part composedCauses : Cause[0..*];
        part composedFailures : Failure[0..*];
        part causalChain : CausalRelationship[0..*];
    }

    part def RiskMetadataRepository {
        attribute repositoryId : String;
        attribute repositoryName : String;
        attribute modelingLanguageCompatibility : String;
        attribute version : String;
        attribute createdDate : String;
        attribute lastModifiedDate : String;
        part allScenarios : Scenario[0..*];
        part allCauses : Cause[0..*];
        part allFailures : Failure[0..*];
        part allCausalRelationships : CausalRelationship[0..*];
        part allRiskScenarios : RiskScenario[0..*];
    }

    port def RiskDataPort {
        attribute portId : String;
        attribute dataFormat : String;
    }

    part def ExternalSystemInterface {
        attribute interfaceId : String;
        attribute interfaceName : String;
        attribute protocolVersion : String;
        port riskDataOut : RiskDataPort;
        port riskDataIn : RiskDataPort;
    }

    part def RiskManagementSystem {
        attribute systemId : String;
        attribute systemName : String;
        attribute systemDescription : String;
        part repository : RiskMetadataRepository[1];
        part externalInterface : ExternalSystemInterface[1];
        port systemRiskDataOut : RiskDataPort;
        port systemRiskDataIn : RiskDataPort;
    }

    part riskSystem : RiskManagementSystem {
        attribute systemId = "RMS-001";
        attribute systemName = "Risk Metadata Management System";
        attribute systemDescription = "Systematic risk metadata modeling and management system";

        part repository : RiskMetadataRepository {
            attribute repositoryId = "REPO-001";
            attribute repositoryName = "Central Risk Repository";
            attribute modelingLanguageCompatibility = "SysML v2";
            attribute version = "1.0.0";

            part allScenarios : Scenario[0..*];
            part allCauses : Cause[0..*];
            part allFailures : Failure[0..*];
            part allCausalRelationships : CausalRelationship[0..*];
            part allRiskScenarios : RiskScenario[0..*];
        }

        part externalInterface : ExternalSystemInterface {
            attribute interfaceId = "EXT-001";
            attribute interfaceName = "Systems Engineering Integration Interface";
            attribute protocolVersion = "2.0";
            port riskDataOut : RiskDataPort;
            port riskDataIn : RiskDataPort;
        }

        port systemRiskDataOut : RiskDataPort;
        port systemRiskDataIn : RiskDataPort;
    }

    connect riskSystem.systemRiskDataOut to riskSystem.externalInterface.riskDataIn;
    connect riskSystem.externalInterface.riskDataOut to riskSystem.systemRiskDataIn;
}