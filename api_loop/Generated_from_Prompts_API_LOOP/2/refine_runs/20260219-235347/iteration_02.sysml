package Vehicles {
    private import ScalarValues::*;

    part def Personnel;
    part def Engine;

    part def Vehicle;

    part def ManualVehicle specializes Vehicle {
        part driver : Personnel[1];
    }

    part def PoweredVehicle specializes Vehicle {
        part engine : Engine[1];
    }

    part def ManualPoweredVehicle specializes ManualVehicle, PoweredVehicle {
    }
}