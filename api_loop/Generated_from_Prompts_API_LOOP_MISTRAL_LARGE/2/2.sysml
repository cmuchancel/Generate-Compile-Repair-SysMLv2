package VehicleSystem {
    private import ScalarValues::*;

    part def Personnel;
    part def Engine;

    part def Vehicle {
        attribute category : VehicleCategory;
    }

    part def ManuallyOperatedVehicle :> Vehicle {
        part driver : Personnel;
    }

    part def PoweredVehicle :> Vehicle {
        part powerSource : Engine;
    }

    part def ManuallyOperatedPoweredVehicle :> ManuallyOperatedVehicle, PoweredVehicle;

    enum def VehicleCategory {
        enum ManuallyOperated;
        enum Powered;
        enum ManuallyOperatedPowered;
    }
}