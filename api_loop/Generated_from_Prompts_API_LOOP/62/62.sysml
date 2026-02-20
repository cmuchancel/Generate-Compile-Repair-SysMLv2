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

        attribute maxMass : Real = 2500;
        attribute totalMass : Real = chassis.mass + engine.mass + gearbox.mass;
        attribute overweightWarning : Boolean = totalMass > maxMass;
    }
}