package VehicleMassManagement {
    private import ScalarValues::*;

    enum def MassStatus {
        compliant;
        overweight;
    }

    part def Chassis {
        attribute mass : ScalarValues::Real;
    }

    part def Engine {
        attribute mass : ScalarValues::Real;
    }

    part def Gearbox {
        attribute mass : ScalarValues::Real;
    }

    part def Vehicle {
        part chassis : Chassis;
        part engine : Engine;
        part gearbox : Gearbox;
        attribute totalMass : ScalarValues::Real;
        attribute status : MassStatus = MassStatus::compliant;
        attribute maxPermittedMass : ScalarValues::Real = 2500.0;

        attribute calculateTotalMass() : ScalarValues::Real {
            chassis.mass + engine.mass + gearbox.mass
        }

        attribute checkMassCompliance() : MassStatus {
            if (calculateTotalMass() > maxPermittedMass) then MassStatus::overweight
            else MassStatus::compliant
            endif
        }
    }
}