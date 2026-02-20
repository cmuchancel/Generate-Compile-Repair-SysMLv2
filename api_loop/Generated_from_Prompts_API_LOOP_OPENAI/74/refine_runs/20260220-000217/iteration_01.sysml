package EngineSelectionAnalysis {
  private import ScalarValues::*;

  enum def EngineType { fourCylinder; sixCylinder; }

  part def Engine {
    attribute type : EngineType;
    attribute power : Real;
    attribute mass : Real;
    attribute efficiency : Real;
    attribute cost : Real;
  }

  part def EngineCandidate {
    part engine : Engine;
    attribute aggregatedPower : Real;
    attribute aggregatedMass : Real;
    attribute aggregatedEfficiency : Real;
    attribute aggregatedCost : Real;
    attribute overallScore : Real;
  }

  part def SelectionSystem {
    part fourCylinderCandidate : EngineCandidate;
    part sixCylinderCandidate : EngineCandidate;
    attribute recommendedType : EngineType;
    attribute bestScore : Real;
  }

  part selectionSystem : SelectionSystem;
}