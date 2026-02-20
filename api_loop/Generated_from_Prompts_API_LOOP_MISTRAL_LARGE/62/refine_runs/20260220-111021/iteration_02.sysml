package VehicleMassManagement {
    private import ScalarValues::*;

    part def Chassis {
        attribute mass : Real;
    }

    part def Engine {
        attribute mass : Real;
    }

    part def Gearbox {
        attribute mass : Real;
    }

    part def Vehicle {
        part chassis : Chassis;
        part engine : Engine;
        part gearbox : Gearbox;
        attribute totalMass : Real;
        attribute maxPermittedMass : Real = 2500.0;
        attribute isOverweight : Boolean;
    }

    item def OverweightWarning;

    part vehicle : Vehicle;

    attribute vehicle.totalMass = vehicle.chassis.mass + vehicle.engine.mass + vehicle.gearbox.mass;
    attribute vehicle.isOverweight = vehicle.totalMass > vehicle.maxPermittedMass;

    connect vehicle.isOverweight to OverweightWarning if vehicle.isOverweight;
}