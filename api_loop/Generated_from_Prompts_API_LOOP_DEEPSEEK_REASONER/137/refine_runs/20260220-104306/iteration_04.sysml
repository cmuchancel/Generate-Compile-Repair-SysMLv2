package AutomotiveComponentModel {
    private import ScalarValues::*;

    enum def WheelPosition {
        enum LEFT;
        enum RIGHT;
    }

    part def Tire {
        attribute height : Real;
    }

    part def Wheel {
        attribute position : WheelPosition;
        attribute hubDiameter : Real;
        attribute width : Real;
        attribute outerDiameter : Real;
        part tire : Tire;
    }

    part def Vehicle {
        attribute mass : Real;
        attribute length : Real;
        part leftFrontWheel : Wheel;
        part rightFrontWheel : Wheel;
    }
}