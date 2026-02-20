package VehicleMassManagement {
    private import ScalarValues::*;

    enum def MassStatus {
        withinLimit
        overweight
    }

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
        attribute status : MassStatus = MassStatus::withinLimit;
        attribute maxPermittedMass : Real = 2500.0;

        constraint massConstraint {
            totalMass == chassis.mass + engine.mass + gearbox.mass
        }

        constraint statusConstraint {
            totalMass > maxPermittedMass => status == MassStatus::overweight
            totalMass <= maxPermittedMass => status == MassStatus::withinLimit
        }
    }
}