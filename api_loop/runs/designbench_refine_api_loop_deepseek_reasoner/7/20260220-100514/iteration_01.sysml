package VehicleSystem {
    private import ScalarValues::*;

    part def Cylinder {
    }

    part def Engine {
        attribute cylinderCount : Integer;
        part cylinders : Cylinder[cylinderCount];
    }

    part def SmallVehicle {
        part engine : Engine {
            cylinderCount = 4;
        }
    }

    part def LargeVehicle {
        part engine : Engine {
            cylinderCount = 6;
        }
    }
}