package VehicleSystem {
    private import ScalarValues::*;

    enum def VehicleSize {
        Small,
        Standard,
        Large
    }

    part def Engine {
        attribute cylinders : Integer[4..6];
    }

    part def Vehicle {
        attribute size : VehicleSize;
        part engine : Engine;
    }

    part def SmallVehicle :> Vehicle {
        attribute size = VehicleSize::Small;
        part engine : Engine {
            attribute cylinders = 4;
        }
    }

    part def StandardVehicle :> Vehicle {
        attribute size = VehicleSize::Standard;
    }

    part def LargeVehicle :> Vehicle {
        attribute size = VehicleSize::Large;
        part engine : Engine {
            attribute cylinders = 6;
        }
    }
}