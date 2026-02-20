package VehicleSystem {
    private import ScalarValues::*;

    enum def VehicleType {
        enum SmallVehicle;
        enum LargeVehicle;
    }

    part def Engine {
        attribute cylinderCount : Integer;
    }

    part def SmallEngine {
        attribute cylinderCount : Integer = 4;
    }

    part def LargeEngine {
        attribute cylinderCount : Integer = 6;
    }

    part def Vehicle {
        attribute vehicleType : VehicleType;
        part engine : Engine;
    }

    part def SmallVehicle {
        attribute vehicleType : VehicleType = VehicleType::SmallVehicle;
        part engine : SmallEngine;
    }

    part def LargeVehicle {
        attribute vehicleType : VehicleType = VehicleType::LargeVehicle;
        part engine : LargeEngine;
    }
}