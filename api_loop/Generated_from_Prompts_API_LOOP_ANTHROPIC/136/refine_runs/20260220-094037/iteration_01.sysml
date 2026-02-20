package AutomotiveComponents {
    private import ScalarValues::*;

    part def Tire {
        attribute manufacturer : String;
        attribute rimDiameter : Real;
        attribute width : Integer;
    }

    part frenchTire : Tire {
        attribute redefines manufacturer = "Michelin";
        attribute redefines rimDiameter = 18.0;
        attribute redefines width = 245;
    }
}