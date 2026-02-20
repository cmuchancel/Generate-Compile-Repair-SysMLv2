package VehicleQualityAnalysis {
    private import ScalarValues::*;

    part def Vehicle {
        attribute totalMass : Real;
        part engine : Engine;
        part transmission : Transmission;
        part frontAxleAssembly : FrontAxleAssembly;
        part rearAxleAssembly : RearAxleAssembly;
    }

    part def Engine {
        attribute mass : Real;
    }

    part def Transmission {
        attribute mass : Real;
    }

    part def FrontAxleAssembly {
        attribute mass : Real;
    }

    part def RearAxleAssembly {
        attribute mass : Real;
    }

    item def QualityRequirement {
        attribute id : String;
        attribute description : String;
    }

    item def AnalysisUseCase {
        attribute id : String;
        attribute description : String;
    }

    item def AnalysisTask {
        attribute id : String;
        attribute description : String;
    }

    item def AnalysisPlan {
        attribute id : String;
        attribute description : String;
    }
}