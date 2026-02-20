package VehicleQualityManagement {
    private import ScalarValues::*;

    enum def QualityStatus {
        enum Nominal;
        enum Warning;
        enum Critical;
    }

    part def Engine {
        attribute mass : Real = 250.0;
        attribute status : QualityStatus = QualityStatus::Nominal;
    }

    part def Transmission {
        attribute mass : Real = 120.0;
        attribute status : QualityStatus = QualityStatus::Nominal;
    }

    part def FrontAxleAssembly {
        attribute mass : Real = 180.0;
        attribute status : QualityStatus = QualityStatus::Nominal;
    }

    part def RearAxleAssembly {
        attribute mass : Real = 200.0;
        attribute status : QualityStatus = QualityStatus::Nominal;
    }

    part def Vehicle {
        attribute totalMass : Real = 1500.0;
        attribute engineMass : Real = 250.0;
        attribute transmissionMass : Real = 120.0;
        attribute frontAxleMass : Real = 180.0;
        attribute rearAxleMass : Real = 200.0;
        attribute status : QualityStatus = QualityStatus::Nominal;

        part engine : Engine;
        part transmission : Transmission;
        part frontAxle : FrontAxleAssembly;
        part rearAxle : RearAxleAssembly;
    }

    part def QualityRequirement {
        attribute requirementId : String = "QR-001";
        attribute description : String = "Vehicle total mass must be within specified limits";
        attribute maxAllowedMass : Real = 2000.0;
        attribute minAllowedMass : Real = 1000.0;
    }

    part def AnalysisUseCase {
        attribute useCaseId : String = "UC-001";
        attribute name : String = "Vehicle Quality Analysis";
        attribute objective : String = "Evaluate vehicle quality attributes";
    }

    part def AnalysisTask {
        attribute taskId : String = "AT-001";
        attribute name : String = "Mass Aggregation Task";
        attribute description : String = "Aggregate component masses to compute total vehicle mass";
        attribute targetMass : Real = 1500.0;
        attribute status : QualityStatus = QualityStatus::Nominal;
    }

    part def AnalysisPlan {
        attribute planId : String = "AP-001";
        attribute name : String = "Vehicle Quality Analysis Plan";
        attribute objective : String = "Comprehensive evaluation of vehicle quality";

        part requirement : QualityRequirement;
        part useCase : AnalysisUseCase;
        part task : AnalysisTask;
    }

    part def QualityManagementSystem {
        attribute systemId : String = "QMS-001";
        attribute name : String = "Vehicle Quality Management System";
        attribute description : String = "System for managing and analyzing vehicle quality attributes";

        part vehicle : Vehicle;
        part analysisPlan : AnalysisPlan;
    }

    port def DataInputPort {
        attribute inputMass : Real = 0.0;
        attribute inputStatus : QualityStatus = QualityStatus::Nominal;
    }

    port def DataOutputPort {
        attribute outputMass : Real = 0.0;
        attribute outputStatus : QualityStatus = QualityStatus::Nominal;
    }

    part def DataChannel {
        port input : DataInputPort;
        port output : DataOutputPort;
    }

    part qms : QualityManagementSystem;
    part channel : DataChannel;

    connect qms to channel;
}