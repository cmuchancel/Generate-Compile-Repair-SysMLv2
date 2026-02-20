package VehicleQualityAnalysis {
    private import ScalarValues::*;

    enum def QualityAttribute {
        enum mass;
        enum durability;
        enum reliability;
        enum performance;
    }

    part def VehicleType;
    part def EngineType;
    part def TransmissionType;
    part def FrontAxleAssemblyType;
    part def RearAxleAssemblyType;
    part def QualityAnalysisRequirementType;
    part def AnalysisUseCaseType;
    part def AnalysisTaskType;
    part def AnalysisPlanType;

    part def VehicleType {
        attribute totalMass : Real;
        part engine : EngineType;
        part transmission : TransmissionType;
        part frontAxle : FrontAxleAssemblyType;
        part rearAxle : RearAxleAssemblyType;
    }

    part def EngineType {
        attribute mass : Real;
        attribute durability : Real;
        attribute reliability : Real;
    }

    part def TransmissionType {
        attribute mass : Real;
        attribute durability : Real;
    }

    part def FrontAxleAssemblyType {
        attribute mass : Real;
        attribute reliability : Real;
    }

    part def RearAxleAssemblyType {
        attribute mass : Real;
        attribute reliability : Real;
    }

    part def QualityAnalysisRequirementType {
        attribute requirementText : String;
        attribute targetAttribute : QualityAttribute;
    }

    part def AnalysisUseCaseType {
        attribute useCaseName : String;
        attribute useCaseObjective : String;
    }

    part def AnalysisTaskType {
        attribute taskName : String;
        attribute status : String;
    }

    part def AnalysisPlanType {
        attribute planName : String;
    }

    part vehicle : VehicleType;
    part engine : EngineType;
    part transmission : TransmissionType;
    part frontAxle : FrontAxleAssemblyType;
    part rearAxle : RearAxleAssemblyType;
    part qualityRequirement : QualityAnalysisRequirementType;
    part analysisUseCase : AnalysisUseCaseType;
    part analysisTask : AnalysisTaskType;
    part analysisPlan : AnalysisPlanType;

    attribute vehicleTotalMass : Real = vehicle.engine.mass + vehicle.transmission.mass + vehicle.frontAxle.mass + vehicle.rearAxle.mass;
}