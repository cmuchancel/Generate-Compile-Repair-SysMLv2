package AutomotiveComponents {
    private import ScalarValues::*;

    part def Tire {
        attribute manufacturer : String;
        attribute rimDiameter : Real;
        attribute width : Integer;
    }

    part def FrenchTire : Tire {
        attribute manufacturer = "Michelin";
        attribute rimDiameter = 18.0;
        attribute width = 245;
    }

    part exampleFrenchTire : FrenchTire;
}