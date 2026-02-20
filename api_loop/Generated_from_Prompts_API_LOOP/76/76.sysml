package MassVerificationPackage {
  private import ScalarValues::*;

  enum def QualificationStatus {
    enum qualified;
    enum unqualified;
  }

  port def MassPort {
    attribute mass : Real;
  }

  part def Vehicle {
    attribute actualMass : Real;
    attribute status : QualificationStatus;
    port massPort : MassPort;
  }

  part def WeighingEquipment {
    attribute measuredMass : Real;
    port massPort : MassPort;
  }

  part def MassVerificationUnit {
    attribute maxAllowedMass : Real = 2500;
    port massPort : MassPort;
    attribute evaluatedStatus : QualificationStatus;
  }

  part def MassVerificationSystem {
    part vehicle : Vehicle;
    part weighingEquipment : WeighingEquipment;
    part verificationUnit : MassVerificationUnit;
    connect vehicle.massPort to weighingEquipment.massPort;
    connect weighingEquipment.massPort to verificationUnit.massPort;
  }
}