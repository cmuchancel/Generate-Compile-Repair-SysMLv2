package VehicleModel {
    private import ScalarValues::*;

    part def Personnel;

    part def Engine;

    part def Vehicle;

    part def ManuallyOperatedVehicle :> Vehicle {
        attribute driver : Personnel;
    }

    part def PoweredVehicle :> Vehicle {
        attribute engine : Engine;
    }

    part def ManuallyOperatedPoweredVehicle :> ManuallyOperatedVehicle, PoweredVehicle {
    }
}