package TireSystem {
    private import ScalarValues::*;

    enum def TireManufacturer {
        enum Michelin;
        enum Bridgestone;
        enum Goodyear;
        enum Continental;
        enum Pirelli;
    }

    part def Vector3D {
        attribute x : Real = 0.0;
        attribute y : Real = 0.0;
        attribute z : Real = 0.0;
    }

    part def Tire {
        attribute manufacturer : TireManufacturer;
        attribute rimDiameterInches : Real;
        attribute tireWidth : Real;
        attribute installationPosition : Vector3D;
    }

    part def TireDatabase {
        part tires : Tire[*];
    }
}