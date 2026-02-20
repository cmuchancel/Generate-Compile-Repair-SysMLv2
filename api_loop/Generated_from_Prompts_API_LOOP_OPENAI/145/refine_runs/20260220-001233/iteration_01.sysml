package TireInfo
    private import ScalarValues::*;

    item def Vector3 {
        attribute x : Real;
        attribute y : Real;
        attribute z : Real;
    }

    part def Tire {
        attribute manufacturerName : String;
        attribute rimDiameterInInches : Real;
        attribute tireWidth : Real;
        attribute position : Vector3 [0..1];
    }

    part def TireCatalog {
        part tires : Tire[*];
    }

    part MichelinExample : Tire {
        attribute manufacturerName = "Michelin";
        attribute rimDiameterInInches = 18.0;
        attribute tireWidth = 245.0;
    }
end package