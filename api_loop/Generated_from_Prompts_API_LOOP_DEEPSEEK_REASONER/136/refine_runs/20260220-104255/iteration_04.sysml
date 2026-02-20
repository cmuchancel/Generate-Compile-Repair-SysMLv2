package AutomotiveComponents {
    private import ScalarValues::*;

    part def Tire {
        attribute manufacturer : String;
        attribute rimDiameter : Real;
        attribute width : Integer;
    }

    part def FrenchTire : Tire;

    part exampleFrenchTire : FrenchTire {
        attribute manufacturer : String = "Michelin";
        attribute rimDiameter : Real = 18.0;
        attribute width : Integer = 245;
    }
}