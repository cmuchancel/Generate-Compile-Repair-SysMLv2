package PowertrainTorqueGeneration {
  private import ScalarValues::*;

  port def TorquePort {
    attribute torqueNm: Real;
  }

  part def TorqueGeneratorFunction {
    port outTorque: TorquePort;
    attribute producedTorqueNm: Real = 100.0;
  }

  part def Engine {
    port inTorque: TorquePort;
    attribute actualTorqueNm: Real = 100.0;
  }

  part def PowertrainSystem {
    part logic: TorqueGeneratorFunction;
    part engine: Engine;
    connect logic.outTorque to engine.inTorque;
    attribute systemTorqueNm: Real = 100.0;
  }
}