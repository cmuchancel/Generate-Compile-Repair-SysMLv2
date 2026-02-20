package PowertrainConfig
{
  private import ScalarValues::*;

  enum def EngineType
  {
    fourCylinder;
    sixCylinder;
    eightCylinder;
  }

  part def Cylinder
  {
    attribute displacement : Real;
    attribute mass : Real;
  }

  part def Piston
  {
    attribute mass : Real;
    attribute material : String;
  }

  part def ConnectingRod
  {
    attribute mass : Real;
    attribute length : Real;
  }

  part def Crankshaft
  {
    attribute mass : Real;
    attribute stroke : Real;
  }

  part def Engine
  {
    attribute engineType : EngineType;
    attribute cylinderCount : Integer;
    attribute totalPower : Real;
    attribute mass : Real;
    attribute efficiency : Real;
    attribute reliability : Real;
    attribute cost : Real;

    part cylinder : Cylinder;
    part piston : Piston;
    part connectingRod : ConnectingRod;
    part crankshaft : Crankshaft;
  }

  part def PerformanceData
  {
    attribute power : Real;
    attribute mass : Real;
    attribute efficiency : Real;
    attribute reliability : Real;
    attribute cost : Real;
  }

  part def EvaluationModel
  {
    attribute powerWeight : Real;
    attribute massWeight : Real;
    attribute efficiencyWeight : Real;
    attribute reliabilityWeight : Real;
    attribute costWeight : Real;
  }

  part def Recommendation
  {
    attribute recommendedEngineType : EngineType;
    attribute recommendedEngineId : String;
    attribute rationale : String;
    attribute transparencyNote : String;
    attribute isOptimal : Boolean;
  }

  part def EngineScheme
  {
    attribute schemeId : String;
    attribute schemeEngineType : EngineType;
    attribute schemeCylinderCount : Integer;
    part engine : Engine;
    part performance : PerformanceData;
    attribute score : Real;
    attribute isFeasible : Boolean;
  }

  part def TradeOffAnalysis
  {
    attribute methodName : String;
    attribute optimalScore : Real;
    part evaluationModel : EvaluationModel;
    part recommendation : Recommendation;
  }

  part def PowertrainConfigurationSystem
  {
    attribute userRequest : String;
    attribute selectedEngineType : EngineType;

    part tradeOff : TradeOffAnalysis;
    part schemeFour : EngineScheme;
    part schemeSix : EngineScheme;
    part schemeEight : EngineScheme;
  }

  part powertrainSystem : PowertrainConfigurationSystem;
}