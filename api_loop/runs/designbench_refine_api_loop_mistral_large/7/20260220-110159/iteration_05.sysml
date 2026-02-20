package VehicleSystem {
    private import ScalarValues::*;

    part def Vehicle;
    part def Engine;
    part def Cylinder;

    part def SmallVehicle :> Vehicle {
        part engine : Engine {
            part cylinders : Cylinder[4];
        }
    }

    part def LargeVehicle :> Vehicle {
        part engine : Engine {
            part cylinders : Cylinder[6];
        }
    }

    attribute def CylinderCount : Integer;
}