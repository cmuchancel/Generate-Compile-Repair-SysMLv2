package VehicleMassModel {
    private import ScalarValues::*;

    part def Component {
        attribute partNumber : String;
        attribute mass : Real;
    }

    part def Engine {
        attribute partNumber : String;
        attribute mass : Real;
    }

    part def Transmission {
        attribute partNumber : String;
        attribute mass : Real;
    }

    part def Body {
        attribute partNumber : String;
        attribute mass : Real;
    }

    part def Vehicle {
        attribute partNumber : String;
        attribute mass : Real;
        attribute totalMass : Real;

        part engine : Engine;
        part transmission : Transmission;
        part body : Body;
    }

    part myVehicle : Vehicle {
        attribute partNumber = "VH-001";
        attribute mass = 500.0;
        attribute totalMass = 1500.0;

        part engine : Engine {
            attribute partNumber = "ENG-001";
            attribute mass = 600.0;
        }

        part transmission : Transmission {
            attribute partNumber = "TRN-001";
            attribute mass = 200.0;
        }

        part body : Body {
            attribute partNumber = "BDY-001";
            attribute mass = 200.0;
        }
    }
}