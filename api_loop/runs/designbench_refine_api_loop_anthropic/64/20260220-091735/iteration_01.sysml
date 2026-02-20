package VehicleMassManagement {
    private import ScalarValues::*;

    enum def ComplianceStatus {
        enum COMPLIANT;
        enum NON_COMPLIANT;
    }

    part def Chassis {
        attribute mass : Real;
    }

    part def Engine {
        attribute mass : Real;
    }

    part def Transmission {
        attribute mass : Real;
    }

    part def Vehicle {
        attribute massLimit : Real = 2500.0;
        attribute totalMass : Real;

        part chassis : Chassis {
            attribute mass : Real = 800.0;
        }

        part engine : Engine {
            attribute mass : Real = 600.0;
        }

        part transmission : Transmission {
            attribute mass : Real = 200.0;
        }

        attribute complianceStatus : ComplianceStatus;
    }

    part def VehicleMassComplianceSystem {
        part vehicle : Vehicle;

        attribute chassisMass : Real = 800.0;
        attribute engineMass : Real = 600.0;
        attribute transmissionMass : Real = 200.0;
        attribute totalMass : Real = 1600.0;
        attribute massLimit : Real = 2500.0;
        attribute isCompliant : Boolean = true;
    }
}