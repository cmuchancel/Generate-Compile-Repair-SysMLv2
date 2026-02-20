package VehicleQualityAnalysis {
    private import ScalarValues::*;

    enum def QualityAttribute {
        enum mass;
        enum durability;
        enum reliability;
        enum performance;
    }

    part def VehicleDef;
    part def EngineDef;
    part def TransmissionDef;
    part def FrontAxleAssemblyDef;
    part def RearAxleAssemblyDef;
    part def QualityAnalysisRequirementDef;
    part def AnalysisUseCaseDef;
    part def AnalysisTaskDef;
    part def AnalysisPlanDef;

    part def VehicleDef {
        attribute totalMass : Real;
        part engine : EngineDef;
        part transmission : TransmissionDef;
        part frontAxle : FrontAxleAssemblyDef;
        part rearAxle : RearAxleAssemblyDef;
    }

    part def EngineDef {
        attribute mass : Real;
        attribute durability : Real;
        attribute reliability : Real;
    }

    part def TransmissionDef {
        attribute mass : Real;
        attribute durability : Real;
    }

    part def FrontAxleAssemblyDef {
        attribute mass : Real;
        attribute reliability : Real;
    }

    part def RearAxleAssemblyDef {
        attribute mass : Real;
        attribute reliability : Real;
    }

    part def QualityAnalysisRequirementDef {
        attribute requirementText : String;
        attribute targetAttribute : QualityAttribute;
    }

    part def AnalysisUseCaseDef {
        attribute useCaseName : String;
        attribute useCaseObjective : String;
    }

    part def AnalysisTaskDef {
        attribute taskName : String;
        attribute status : String;
    }

    part def AnalysisPlanDef {
        attribute planName : String;
    }

    part vehicle : VehicleDef;
    part engine : EngineDef;
    part transmission : TransmissionDef;
    part frontAxle : FrontAxleAssemblyDef;
    part rearAxle : RearAxleAssemblyDef;
    part requirement : QualityAnalysisRequirementDef;
    part useCase : AnalysisUseCaseDef;
    part task : AnalysisTaskDef;
    part plan : AnalysisPlanDef;

    attribute vehicleTotalMass : Real = vehicle.engine.mass + vehicle.transmission.mass + vehicle.frontAxle.mass + vehicle.rearAxle.mass;
}