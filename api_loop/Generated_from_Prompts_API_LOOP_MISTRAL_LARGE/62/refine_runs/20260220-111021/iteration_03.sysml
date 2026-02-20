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
        attribute calculateTotalMass : Real = chassis.mass + engine.mass + gearbox.mass;
        attribute checkOverweight : Boolean = calculateTotalMass > maxPermittedMass;
    }

    port def OverweightWarningPort {
        attribute warning : Boolean;
    }

    part vehicle : Vehicle {
        attribute totalMass : Real = calculateTotalMass;
        attribute isOverweight : Boolean = checkOverweight;
    }

    part warningSystem {
        port warningPort : OverweightWarningPort;
    }

    connect vehicle.isOverweight to warningSystem.warningPort.warning;
}