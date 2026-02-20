package EngineSystem {
    private import ScalarValues::*;

    enum def CylinderCount {
        enum FOUR;
        enum SIX;
    }

    enum def CylinderDiameter {
        enum SMALL;
        enum LARGE;
    }

    part def Cylinder {
        attribute diameter : Real;
    }

    part def Engine {
        attribute cylinderCount : Integer;
        attribute cylinderDiameter : Real;
    }

    part def FourCylinderEngine {
        attribute cylinderCount : Integer = 4;
        attribute cylinderDiameter : Real;
        part cyl1 : Cylinder;
        part cyl2 : Cylinder;
        part cyl3 : Cylinder;
        part cyl4 : Cylinder;
    }

    part def SixCylinderEngine {
        attribute cylinderCount : Integer = 6;
        attribute cylinderDiameter : Real;
        part cyl1 : Cylinder;
        part cyl2 : Cylinder;
        part cyl3 : Cylinder;
        part cyl4 : Cylinder;
        part cyl5 : Cylinder;
        part cyl6 : Cylinder;
    }

    part engineVariant70mm : FourCylinderEngine {
        attribute redefines cylinderDiameter = 70.0;
    }

    part engineVariant100mm : FourCylinderEngine {
        attribute redefines cylinderDiameter = 100.0;
    }

    part sixCylEngine70mm : SixCylinderEngine {
        attribute redefines cylinderDiameter = 70.0;
    }

    part sixCylEngine100mm : SixCylinderEngine {
        attribute redefines cylinderDiameter = 100.0;
    }
}