package TireSystem {
    private import ScalarValues::*;
    
    part def CoordinateVector {
        attribute x : Real;
        attribute y : Real;
        attribute z : Real;
    }
    
    part def Tire {
        attribute manufacturer : String;
        attribute rimDiameter : Real;
        attribute width : Integer;
        attribute position : CoordinateVector[0..1];
    }
    
    part exampleTire : Tire {
        manufacturer = "Michelin";
        rimDiameter = 18.0;
        width = 245;
    }
}