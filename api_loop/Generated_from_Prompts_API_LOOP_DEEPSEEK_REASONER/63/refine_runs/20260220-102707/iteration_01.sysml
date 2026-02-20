package VehicleMassBudget {
    private import ScalarValues::*;

    part def Chassis {
        attribute mass : Real;
    }

    part def Engine {
        attribute mass : Real;
    }

    part def Transmission {
        attribute mass : Real;
    }

    part def Vehicle {
        part chassis : Chassis;
        part engine : Engine;
        part transmission : Transmission;
        attribute totalMass : Real;
        attribute massLimit : Real = 2500;
        item def MassConstraint {
            attribute requirementMet : Boolean;
        }
        part massCheck : MassConstraint;
    }

    part massBudgetSystem : Vehicle;
}