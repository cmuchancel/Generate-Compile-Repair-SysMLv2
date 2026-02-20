package AutomotiveComponents {
    private import ScalarValues::*;

    part def Tire {
        attribute manufacturer : string;
        attribute rimDiameter : Real;
        attribute width : Integer;
    }

    part exampleFrenchTire : Tire {
        attribute manufacturer = "Michelin";
        attribute rimDiameter = 18.0;
        attribute width = 245;
    }
}