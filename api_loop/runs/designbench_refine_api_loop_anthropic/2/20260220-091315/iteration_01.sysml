package VehicleSystem {
    private import ScalarValues::*;

    part def Personnel;
    part def Engine;

    part def Vehicle;

    part def ManuallyOperatedVehicle :> Vehicle {
        part driver : Personnel[1];
    }

    part def PoweredVehicle :> Vehicle {
        part engine : Engine[1];
    }

    part def ManuallyOperatedPoweredVehicle :> ManuallyOperatedVehicle, PoweredVehicle;

    part manuallyOperatedVehicle : ManuallyOperatedVehicle {
        part driver : Personnel[1];
    }

    part poweredVehicle : PoweredVehicle {
        part engine : Engine[1];
    }

    part manuallyOperatedPoweredVehicle : ManuallyOperatedPoweredVehicle {
        part driver : Personnel[1];
        part engine : Engine[1];
    }
}