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
        attribute x : Real;
        attribute y : Real;
        attribute z : Real;
    }

    part def Tire {
        attribute manufacturer : TireManufacturer;
        attribute rimDiameterInches : Real;
        attribute tireWidth : Real;
        attribute installationPosition : Vector3D = (x: 0.0, y: 0.0, z: 0.0);
    }

    part def TireDatabase {
        part tires : Tire[*];
    }
}