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

  part def Engine {
    attribute mass : Real;
    attribute quality : QualityData;
  }

  part def Transmission {
    attribute mass : Real;
    attribute quality : QualityData;
  }

  part def FrontAxleAssembly {
    attribute mass : Real;
    attribute quality : QualityData;
  }

  part def RearAxleAssembly {
    attribute mass : Real;
    attribute quality : QualityData;
  }

  part def Vehicle {
    part engine : Engine;
    part transmission : Transmission;
    part frontAxle : FrontAxleAssembly;
    part rearAxle : RearAxleAssembly;
    attribute totalMass : Real = engine.mass + transmission.mass + frontAxle.mass + rearAxle.mass;
  }

  part def QualityAnalysisRequirement {
    attribute objectiveText : String;
    attribute threshold : Real;
    attribute status : QualityStatus;
  }

  part def AnalysisUseCase {
    attribute useCaseName : String;
    attribute goalText : String;
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
    part qaRequirement : QualityAnalysisRequirement;
    part analysisUseCase : AnalysisUseCase;
    part analysisTask : AnalysisTask;
    part analysisPlan : AnalysisPlan;
  }
}