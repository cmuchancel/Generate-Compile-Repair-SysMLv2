package AutomotiveSystem {
    private import ScalarValues::*;

    attribute def MassValue : Real;
    attribute def LengthValue : Real;

    part def Car {
        attribute totalMass : MassValue = 1200.0;
        attribute length : LengthValue = 4.82;
        part wheelAssemblies : WheelAssembly[4];
    }

    part def WheelAssembly {
        part wheel : Wheel;
        part brakeDisc : BrakeDisc;
    }

    part def Wheel {
        part rim : Rim;
        part tire : Tire;
    }

    part def Rim {
        attribute diameter : LengthValue = 18.0;
    }

    part def Tire {
        attribute width : LengthValue = 245.0;
        attribute height : LengthValue = 45.0;
        attribute treadDepth : LengthValue = 6.0;
        attribute minTreadDepth : LengthValue = 3.5;
        attribute outerDiameter : LengthValue = (rim.diameter * 25.4) + (2 * height);
    }

    part def BrakeDisc {
        attribute radius : LengthValue = 95.0;
        attribute diameter : LengthValue = 2 * radius;
    }

    part car : Car;

    constraint def WheelAssemblyFitment {
        car.wheelAssemblies.brakeDisc.diameter < car.wheelAssemblies.wheel.tire.outerDiameter;
    }

    constraint def TreadDepthCompliance {
        car.wheelAssemblies.wheel.tire.treadDepth >= car.wheelAssemblies.wheel.tire.minTreadDepth;
    }
}