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
        attribute outerDiameter : LengthValue = (rim.diameter * 25.4 mm) + (2 * (width * height / 100.0));
    }

    part def BrakeDisc {
        attribute radius : LengthValue = 95.0 mm;
        attribute diameter : LengthValue = radius * 2;
    }

    requirement TreadDepthRequirement {
        attribute description : String = "Tire tread depth must not be less than 3.5 millimeters.";
        attribute satisfied : Boolean = tire.treadDepth >= tire.minTreadDepth;
    }

    requirement BrakeDiscFitmentRequirement {
        attribute description : String = "Brake disc diameter must be less than wheel outer diameter.";
        attribute satisfied : Boolean = brakeDisc.diameter < wheel.outerDiameter;
    }

    part car : Car;
}