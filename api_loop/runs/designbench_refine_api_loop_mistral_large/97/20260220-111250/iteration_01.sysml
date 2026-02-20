package VehicleSystem {
    private import ScalarValues::*;

    enum def VehicleSize {
        enum Small;
        enum Standard;
        enum Large;
    }

    part def Cylinder;

    part def Engine {
        attribute cylinderCount : Integer;
        part cylinders : Cylinder[4..6];
    }

    part def Vehicle {
        attribute size : VehicleSize;
        part engine : Engine;

        constraint {
            if size == VehicleSize::Small then engine.cylinderCount == 4
            else if size == VehicleSize::Large then engine.cylinderCount == 6
            else engine.cylinderCount >= 4 and engine.cylinderCount <= 6
        }
    }
}