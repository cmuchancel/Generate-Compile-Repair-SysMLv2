package VehicleSystem {
    private import ScalarValues::*;

    part def Cylinder;

    part def Engine {
        part cylinders : Cylinder;
        attribute cylinderCount : Integer;
    }

    part def SmallEngine specializes Engine {
        attribute cylinderCount : Integer = 4;
    }

    part def LargeEngine specializes Engine {
        attribute cylinderCount : Integer = 6;
    }

    part def Vehicle {
        part engine : Engine;
    }

    part def SmallVehicle specializes Vehicle {
        part engine : SmallEngine;
    }

    part def LargeVehicle specializes Vehicle {
        part engine : LargeEngine;
    }
}