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

        attribute smallConstraint : Boolean = size == VehicleSize::Small implies engine.cylinderCount == 4;
        attribute largeConstraint : Boolean = size == VehicleSize::Large implies engine.cylinderCount == 6;
        attribute standardConstraint : Boolean = size == VehicleSize::Standard implies (engine.cylinderCount >= 4 and engine.cylinderCount <= 6);
    }
}