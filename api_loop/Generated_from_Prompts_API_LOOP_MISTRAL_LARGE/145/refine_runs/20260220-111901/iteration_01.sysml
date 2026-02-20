package TireSystem {
    private import ScalarValues::*;

    enum def TireManufacturer {
        enum Michelin;
        enum Bridgestone;
        enum Goodyear;
        enum Continental;
        enum Pirelli;
    }

    part def Tire {
        attribute manufacturer : TireManufacturer;
        attribute rimDiameterInches : Real;
        attribute tireWidth : Real;
        attribute installationPosition : Real[3] = [0.0, 0.0, 0.0];
    }

    part def TireDatabase {
        part tires : Tire[*];
    }
}