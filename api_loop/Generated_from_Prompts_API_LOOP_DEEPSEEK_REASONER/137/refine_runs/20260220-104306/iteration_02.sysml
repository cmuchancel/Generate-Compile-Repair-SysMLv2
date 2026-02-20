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
        attribute hubDiameter : Length = in(18.0);
        attribute width : Length = mm(245.0);
        attribute outerDiameter : Length = hubDiameter + 2 * tire.height;
        part tire : Tire;
    }

    part def Vehicle {
        attribute mass : Mass = kg(1200.0);
        attribute length : Length = m(4.82);
        part leftFrontWheel : Wheel {
            position = WheelPosition::LEFT;
        }
        part rightFrontWheel : Wheel {
            position = WheelPosition::RIGHT;
        }
    }
}