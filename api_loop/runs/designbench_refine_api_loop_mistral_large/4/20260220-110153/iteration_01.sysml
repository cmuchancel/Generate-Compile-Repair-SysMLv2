package VehicleSystem {
    private import ScalarValues::*;

    enum def VehicleType {
        enum SmallVehicle;
        enum LargeVehicle;
    }

    part def Engine {
        attribute cylinderCount : Integer;
    }

    part def SmallEngine : Engine {
        attribute cylinderCount : Integer = 4;
    }

    part def LargeEngine : Engine {
        attribute cylinderCount : Integer = 6;
    }

    part def Vehicle {
        attribute vehicleType : VehicleType;
        part engine : Engine;
    }

    part def SmallVehicle : Vehicle {
        attribute vehicleType : VehicleType = VehicleType::SmallVehicle;
        part engine : SmallEngine;
    }

    part def LargeVehicle : Vehicle {
        attribute vehicleType : VehicleType = VehicleType::LargeVehicle;
        part engine : LargeEngine;
    }
}