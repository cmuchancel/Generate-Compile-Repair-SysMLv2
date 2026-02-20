package AutomotiveSystem {
    private import ScalarValues::*;
    
    part def Vehicle {
        attribute totalMass : Mass = 1200.0::kg;
        attribute length : Length = 4.82::m;
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
        attribute outerDiameter : Length = rim.diameter + 2 * tire.height;
    }
    
    part def Rim {
        attribute diameter : Length = 457.2::mm; // 18 inches converted to mm
    }
    
    part def Tire {
        attribute width : Length = 245.0::mm;
        attribute height : Length = 45.0::mm;
        attribute treadDepth : Length = 6.0::mm;
        attribute minTreadDepth : Length = 3.5::mm;
        attribute isTreadSafe : Boolean = treadDepth >= minTreadDepth;
    }
    
    part def BrakeDisc {
        attribute radius : Length = 95.0::mm;
        attribute diameter : Length = 2 * radius;
    }
}