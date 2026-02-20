package EnginePackage {
    private import ScalarValues::*;

    enum def CylinderCount {
        enum four,
        enum six
    }

    enum def CylinderDiameter {
        enum seventy_mm : Real = 70.0,
        enum one_hundred_mm : Real = 100.0
    }

    part def Cylinder {
        attribute diameter : CylinderDiameter;
    }

    part def Engine {
        attribute cylinderCount : CylinderCount;
        part cylinders : Cylinder[0..*];
    }

    part def FourCylinderEngine {
        attribute cylinderCount : CylinderCount = CylinderCount::four;
        part cylinders : Cylinder[4];
    }

    part def SixCylinderEngine {
        attribute cylinderCount : CylinderCount = CylinderCount::six;
        part cylinders : Cylinder[6];
    }
}