package EngineConfig {
  private import ScalarValues::*;

  enum def CylinderDiameter { d70; d100; }
  enum def CylinderCount { four; six; }

  part def Cylinder {
    attribute diameter : CylinderDiameter;
  }

  part def ConfigurableEngine {
    attribute cylinderCount : CylinderCount;
    attribute cylinderDiameter : CylinderDiameter;
  }

  part def FourCylinderEngine {
    part cyl1 : Cylinder;
    part cyl2 : Cylinder;
    part cyl3 : Cylinder;
    part cyl4 : Cylinder;
    attribute cylinderCount : CylinderCount = CylinderCount::four;
    attribute cylinderDiameter : CylinderDiameter;
  }

  part def SixCylinderEngine {
    part cyl1 : Cylinder;
    part cyl2 : Cylinder;
    part cyl3 : Cylinder;
    part cyl4 : Cylinder;
    part cyl5 : Cylinder;
    part cyl6 : Cylinder;
    attribute cylinderCount : CylinderCount = CylinderCount::six;
    attribute cylinderDiameter : CylinderDiameter;
  }

  part configurableEngine : ConfigurableEngine;
  part fourCylinderEngine : FourCylinderEngine;
  part sixCylinderEngine : SixCylinderEngine;
}