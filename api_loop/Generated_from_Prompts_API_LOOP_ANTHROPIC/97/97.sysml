package VehicleSystem {
    private import ScalarValues::*;

    enum def VehicleSize {
        enum small;
        enum large;
    }

    part def Cylinder;

    part def Engine {
        part cylinders : Cylinder[4..6];
    }

    part def SmallEngine {
        part cylinders : Cylinder[4];
    }

    part def LargeEngine {
        part cylinders : Cylinder[6];
    }

    part def Vehicle {
        attribute size : VehicleSize;
        part engine : Engine[1];
    }

    part def SmallVehicle {
        attribute size : VehicleSize = VehicleSize::small;
        part engine : SmallEngine[1];
    }

    part def LargeVehicle {
        attribute size : VehicleSize = VehicleSize::large;
        part engine : LargeEngine[1];
    }

    part smallVehicleInstance : SmallVehicle {
        attribute size : VehicleSize = VehicleSize::small;
        part engine : SmallEngine[1] {
            part cylinders : Cylinder[4];
        }
    }

    part largeVehicleInstance : LargeVehicle {
        attribute size : VehicleSize = VehicleSize::large;
        part engine : LargeEngine[1] {
            part cylinders : Cylinder[6];
        }
    }
}