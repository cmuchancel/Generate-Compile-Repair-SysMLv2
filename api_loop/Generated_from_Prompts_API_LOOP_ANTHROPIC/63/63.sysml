package VehicleMassBudget {
    private import ScalarValues::*;

    part def Chassis {
        attribute actualMass : Real = 800.0;
    }

    part def Engine {
        attribute actualMass : Real = 450.0;
    }

    part def Transmission {
        attribute actualMass : Real = 120.0;
    }

    part def Vehicle {
        attribute massLimit : Real = 2500.0;
        attribute totalMass : Real = 1370.0;
        attribute massWithinLimit : Boolean = true;

        part chassis : Chassis;
        part engine : Engine;
        part transmission : Transmission;
    }

    part vehicle : Vehicle;
}