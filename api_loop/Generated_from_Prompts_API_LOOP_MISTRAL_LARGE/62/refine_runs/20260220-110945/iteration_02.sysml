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
        attribute status : MassStatus;
        attribute maxPermittedMass : Real = 2500.0;

        attribute calculateTotalMass : Real = chassis.mass + engine.mass + gearbox.mass;

        attribute checkMassCompliance : MassStatus =
            if (totalMass > maxPermittedMass) then MassStatus::overweight
            else MassStatus::withinLimit
            endif;
    }
}