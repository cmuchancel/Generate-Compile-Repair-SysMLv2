package TireInformationSystem {
    private import ScalarValues::*;

    enum def Manufacturer {
        enum Michelin;
        enum Bridgestone;
        enum Goodyear;
        enum Continental;
        enum Pirelli;
    }

    part def CoordinateVector {
        attribute x : Real;
        attribute y : Real;
        attribute z : Real;
    }

    part def InstallationPosition {
        part location : CoordinateVector;
    }

    part def Tire {
        attribute manufacturerName : Manufacturer;
        attribute rimDiameterInches : Real;
        attribute tireWidth : Real;
        part installationPosition : InstallationPosition;
    }

    part def TireCollection {
        part michelinTire18x245 : Tire {
            attribute manufacturerName : Manufacturer = Manufacturer::Michelin;
            attribute rimDiameterInches : Real = 18.0;
            attribute tireWidth : Real = 245.0;
            part installationPosition : InstallationPosition {
                part location : CoordinateVector {
                    attribute x : Real = 0.0;
                    attribute y : Real = 0.0;
                    attribute z : Real = 0.0;
                }
            }
        }

        part bridgestoneTire17x225 : Tire {
            attribute manufacturerName : Manufacturer = Manufacturer::Bridgestone;
            attribute rimDiameterInches : Real = 17.0;
            attribute tireWidth : Real = 225.0;
            part installationPosition : InstallationPosition {
                part location : CoordinateVector {
                    attribute x : Real = 1.5;
                    attribute y : Real = 0.0;
                    attribute z : Real = 0.0;
                }
            }
        }

        part continentalTire19x255 : Tire {
            attribute manufacturerName : Manufacturer = Manufacturer::Continental;
            attribute rimDiameterInches : Real = 19.0;
            attribute tireWidth : Real = 255.0;
        }
    }
}