package VehicleMassManagement {
    private import ScalarValues::*;

    enum def MassStatus {
        enum OK;
        enum OVERWEIGHT;
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
        attribute maxPermittedMass : Real = 2500.0;

        part chassis : Chassis {
            attribute mass : Real = 1200.0;
        }

        part engine : Engine {
            attribute mass : Real = 800.0;
        }

        part gearbox : Gearbox {
            attribute mass : Real = 300.0;
        }

        attribute totalMass : Real = 1200.0 + 800.0 + 300.0;

        attribute status : MassStatus = MassStatus::OK;
    }

    part vehicle : Vehicle;
}