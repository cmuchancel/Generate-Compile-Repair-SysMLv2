package AutomotiveComponents {
    private import ScalarValues::*;

    part def Tire {
        attribute manufacturer : String;
        attribute rimDiameter : Real;
        attribute width : Integer;
    }

    part michelinTire : Tire {
        manufacturer = "Michelin";
        rimDiameter = 18.0;
        width = 245;
    }
}