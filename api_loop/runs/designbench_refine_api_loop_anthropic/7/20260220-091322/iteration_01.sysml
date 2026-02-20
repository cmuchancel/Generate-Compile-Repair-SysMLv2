package VehicleSystem {
    private import ScalarValues::*;

    part def Cylinder;

    part def Engine {
        part cylinders : Cylinder[4..6];
    }

    part def SmallVehicleEngine {
        part cylinders : Cylinder[4];
    }

    part def LargeVehicleEngine {
        part cylinders : Cylinder[6];
    }

    part def SmallVehicle {
        part engine : SmallVehicleEngine[1];
    }

    part def LargeVehicle {
        part engine : LargeVehicleEngine[1];
    }

    part smallVehicle : SmallVehicle {
        part redefines engine : SmallVehicleEngine {
            part redefines cylinders : Cylinder[4];
        }
    }

    part largeVehicle : LargeVehicle {
        part redefines engine : LargeVehicleEngine {
            part redefines cylinders : Cylinder[6];
        }
    }
}