package AutomotiveSystem {
    private import ScalarValues::*;

    part def Vehicle {
        attribute totalMass : Real = 1200.0;
        attribute length : Real = 4.82;
        part wheelAssemblies : WheelAssembly[4];
    }

    part def WheelAssembly {
        part wheel : Wheel;
        part brakeDisc : BrakeDisc;
        attribute isBrakeDiscFitting : Boolean = brakeDisc.diameter < wheel.outerDiameter;
    }

    part def Wheel {
        part rim : Rim;
        part tire : Tire;
        attribute outerDiameter : Real = rim.diameter + 2 * tire.height;
    }

    part def Rim {
        attribute diameter : Real = 457.2;
    }

    part def Tire {
        attribute width : Real = 245.0;
        attribute height : Real = 45.0;
        attribute treadDepth : Real = 6.0;
        attribute minTreadDepth : Real = 3.5;
        attribute isTreadSafe : Boolean = treadDepth >= minTreadDepth;
    }

    part def BrakeDisc {
        attribute radius : Real = 95.0;
        attribute diameter : Real = 2 * radius;
    }
}