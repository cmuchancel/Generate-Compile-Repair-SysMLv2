package AutomotiveComponentModel {

    private import ScalarValues::*;

    enum def WheelPosition {
        enum LEFT;
        enum RIGHT;
    }

    part def Tire {
    }

    part def Wheel {
        attribute position : WheelPosition;
        attribute hubDiameter : Length = 18.0::in;
        attribute width : Length = 245.0::mm;
        part tire : Tire;
    }

    part def Vehicle {
        attribute mass : Mass = 1200.0::kg;
        attribute length : Length = 4.82::m;
        part leftFrontWheel : Wheel {
            position = WheelPosition::LEFT;
        }
        part rightFrontWheel : Wheel {
            position = WheelPosition::RIGHT;
        }
    }
}