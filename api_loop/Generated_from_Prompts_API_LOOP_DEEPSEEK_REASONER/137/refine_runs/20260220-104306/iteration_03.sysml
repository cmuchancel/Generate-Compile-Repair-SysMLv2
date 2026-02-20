package AutomotiveComponentModel {
    private import ScalarValues::*;

    enum def WheelPosition {
        enum LEFT;
        enum RIGHT;
    }

    part def Tire {
        attribute height : Length;
    }

    part def Wheel {
        attribute position : WheelPosition;
        attribute hubDiameter : Length = 18.0[inch];
        attribute width : Length = 245.0[mm];
        attribute outerDiameter : Length;
        part tire : Tire;
    }

    part def Vehicle {
        attribute mass : Mass = 1200.0[kg];
        attribute length : Length = 4.82[m];
        part leftFrontWheel : Wheel;
        part rightFrontWheel : Wheel;
    }
}