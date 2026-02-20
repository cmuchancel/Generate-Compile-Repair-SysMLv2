package AutomotiveSystem {
    private import ScalarValues::*;

    part def Car {
        attribute totalMass : MassValue = 1200.0 kg;
        attribute length : LengthValue = 4.82 m;
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
        attribute diameter : LengthValue = 18.0 in;
    }

    part def Tire {
        attribute width : LengthValue = 245.0 mm;
        attribute height : LengthValue = 45.0 mm;
        attribute treadDepth : LengthValue = 6.0 mm;
        attribute minTreadDepth : LengthValue = 3.5 mm;
        attribute outerDiameter : LengthValue;
    }

    part def BrakeDisc {
        attribute radius : LengthValue = 95.0 mm;
        attribute diameter : LengthValue;
    }

    attribute def MassValue : Real;
    attribute def LengthValue : Real;

    constraint def WheelAssemblyFitment {
        wheelAssembly : WheelAssembly;
        wheelAssembly.brakeDisc.diameter < wheelAssembly.wheel.tire.outerDiameter;
    }

    constraint def TreadDepthCompliance {
        wheelAssembly : WheelAssembly;
        wheelAssembly.wheel.tire.treadDepth >= wheelAssembly.wheel.tire.minTreadDepth;
    }

    part car : Car;

    // Calculate outer diameter of tire (simplified)
    car.wheelAssemblies.tire.outerDiameter = (car.wheelAssemblies.wheel.rim.diameter * 25.4 mm) + (2 * car.wheelAssemblies.tire.height);

    // Calculate brake disc diameter
    car.wheelAssemblies.brakeDisc.diameter = 2 * car.wheelAssemblies.brakeDisc.radius;
}