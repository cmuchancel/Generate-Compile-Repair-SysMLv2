package VehicleSystem {
    private import ScalarValues::*;

    part def Engine {
        attribute cylinders : Integer;
    }

    part def SmallEngine :> Engine {
        attribute cylinders : Integer = 4;
    }

    part def LargeEngine :> Engine {
        attribute cylinders : Integer = 6;
    }

    part def Vehicle {
        part engine : Engine;
    }

    part def SmallVehicle :> Vehicle {
        part engine : SmallEngine;
    }

    part def LargeVehicle :> Vehicle {
        part engine : LargeEngine;
    }
}