package VehicleQualityAnalysis {
  private import ScalarValues::*;

  enum def QualityStatus {
    enum GOOD;
    enum WARNING;
    enum FAILED;
  }

  item def QualityData {
    attribute defectCount : Integer;
    attribute inspectionScore : Real;
    attribute status : QualityStatus;
  }

  part def Component {
    attribute mass : Real;
    attribute quality : QualityData;
  }

  part def Engine : Component;

  part def Transmission : Component;

  part def AxleAssembly : Component;

  part def FrontAxleAssembly : AxleAssembly;

  part def RearAxleAssembly : AxleAssembly;

  part def Vehicle {
    part engine : Engine;
    part transmission : Transmission;
    part frontAxle : FrontAxleAssembly;
    part rearAxle : RearAxleAssembly;
    attribute totalMass : Real = engine.mass + transmission.mass + frontAxle.mass + rearAxle.mass;
  }

  part def QualityAnalysisRequirement {
    attribute objective : String;
    attribute threshold : Real;
    attribute status : QualityStatus;
  }

  part def AnalysisUseCase {
    attribute name : String;
    attribute goal : String;
  }

  part def AnalysisTask {
    attribute taskName : String;
    attribute description : String;
    attribute owner : String;
  }

  part def AnalysisPlan {
    attribute planName : String;
    part task : AnalysisTask;
  }

  part def QualityAnalysisSystem {
    part vehicle : Vehicle;
    part requirement : QualityAnalysisRequirement;
    part useCase : AnalysisUseCase;
    part task : AnalysisTask;
    part plan : AnalysisPlan;
  }
}