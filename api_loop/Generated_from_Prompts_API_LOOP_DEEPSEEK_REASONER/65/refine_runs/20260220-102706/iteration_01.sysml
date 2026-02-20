package VehicleWeightManagementSystem {
    private import ScalarValues::*;

    part def Chassis {
        attribute weight : Real;
    }

    part def Engine {
        attribute weight : Real;
    }

    part def Transmission {
        attribute weight : Real;
    }

    part def Vehicle {
        attribute totalWeight : Real;
        attribute weightLimit : Real = 2500.0;
        part chassis : Chassis;
        part engine : Engine;
        part transmission : Transmission;
    }
}