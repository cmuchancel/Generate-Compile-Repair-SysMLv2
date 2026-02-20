package VehicleMassManagement {
    private import ScalarValues::*;

    enum def MassStatus {
        compliant;
        overweight;
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
        attribute status : MassStatus = MassStatus::compliant;
        attribute maxPermittedMass : Real = 2500.0;

        attribute calculateTotalMass() : Real {
            chassis.mass + engine.mass + gearbox.mass
        }

        attribute checkMassCompliance() : MassStatus {
            if (calculateTotalMass() > maxPermittedMass) then MassStatus::overweight
            else MassStatus::compliant
            endif
        }
    }
}