package Vehicles {
    private import ScalarValues::*;

    part def Personnel;
    part def Engine;

    part def Vehicle;

    part def ManualVehicle : Vehicle {
        part driver : Personnel[1];
    }

    part def PoweredVehicle : Vehicle {
        part engine : Engine[1];
    }

    part def ManualPoweredVehicle : ManualVehicle, PoweredVehicle {
    }
}