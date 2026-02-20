package AutomotiveSystem {
    private import ScalarValues::*;

    part def Car {
        attribute totalMass : Real = 1200.0;
        attribute length : Real = 4.82;
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
        attribute diameterInches : Real = 18.0;
        attribute diameter : Real = diameterInches * 25.4;
    }

    part def Tire {
        attribute width : Real = 245.0;
        attribute height : Real = 45.0;
        attribute treadDepth : Real = 6.0;
        attribute minTreadDepth : Real = 3.5;
        attribute outerDiameter : Real = (rim.diameter) + (2 * (width * height / 100.0));
    }

    part def BrakeDisc {
        attribute radius : Real = 95.0;
        attribute diameter : Real = radius * 2;
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
    part carWheelAssembly : WheelAssembly;
    part carWheel : Wheel;
    part carRim : Rim;
    part carTire : Tire;
    part carBrakeDisc : BrakeDisc;
}