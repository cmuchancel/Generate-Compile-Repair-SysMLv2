package VehicleQualityAnalysis {
    private import ScalarValues::*;

    enum def QualityAttribute {
        mass,
        durability,
        reliability,
        performance
    }

    part def Vehicle;
    part def Engine;
    part def Transmission;
    part def FrontAxleAssembly;
    part def RearAxleAssembly;
    part def QualityAnalysisRequirement;
    part def AnalysisUseCase;
    part def AnalysisTask;
    part def AnalysisPlan;

    part def Vehicle {
        attribute totalMass : Real;
        part engine : Engine;
        part transmission : Transmission;
        part frontAxle : FrontAxleAssembly;
        part rearAxle : RearAxleAssembly;
    }

    part def Engine {
        attribute mass : Real;
        attribute durability : Real;
        attribute reliability : Real;
    }

    part def Transmission {
        attribute mass : Real;
        attribute durability : Real;
    }

    part def FrontAxleAssembly {
        attribute mass : Real;
        attribute reliability : Real;
    }

    part def RearAxleAssembly {
        attribute mass : Real;
        attribute reliability : Real;
    }

    part def QualityAnalysisRequirement {
        attribute requirementText : String;
        attribute targetAttribute : QualityAttribute;
    }

    part def AnalysisUseCase {
        attribute useCaseName : String;
        attribute objective : String;
    }

    part def AnalysisTask {
        attribute taskName : String;
        attribute status : String;
    }

    part def AnalysisPlan {
        attribute planName : String;
        attribute tasks : AnalysisTask[0..*];
    }

    part vehicle : Vehicle;
    part engine : Engine;
    part transmission : Transmission;
    part frontAxle : FrontAxleAssembly;
    part rearAxle : RearAxleAssembly;
    part requirement : QualityAnalysisRequirement;
    part useCase : AnalysisUseCase;
    part task : AnalysisTask;
    part plan : AnalysisPlan;

    attribute vehicle.totalMass : Real = vehicle.engine.mass + vehicle.transmission.mass + vehicle.frontAxle.mass + vehicle.rearAxle.mass;
}