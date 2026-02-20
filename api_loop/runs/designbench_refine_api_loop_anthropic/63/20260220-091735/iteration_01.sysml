package VehicleMassBudget {
    private import ScalarValues::*;

    attribute def MassValue;

    part def Chassis {
        attribute actualMass : Real;
    }

    part def Engine {
        attribute actualMass : Real;
    }

    part def Transmission {
        attribute actualMass : Real;
    }

    part def Vehicle {
        attribute massLimit : Real = 2500.0;
        attribute totalMass : Real;
        attribute massWithinLimit : Boolean;

        part chassis : Chassis {
            attribute actualMass : Real = 800.0;
        }

        part engine : Engine {
            attribute actualMass : Real = 450.0;
        }

        part transmission : Transmission {
            attribute actualMass : Real = 120.0;
        }
    }

    part vehicle : Vehicle {
        attribute redefines massLimit = 2500.0;
        attribute redefines totalMass = 1370.0;
        attribute redefines massWithinLimit = true;

        part redefines chassis {
            attribute redefines actualMass = 800.0;
        }

        part redefines engine {
            attribute redefines actualMass = 450.0;
        }

        part redefines transmission {
            attribute redefines actualMass = 120.0;
        }
    }
}