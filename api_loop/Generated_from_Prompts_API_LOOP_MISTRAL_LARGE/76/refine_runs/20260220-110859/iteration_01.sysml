package VehicleMassVerification {
    private import ScalarValues::*;

    enum def QualificationStatus {
        enum qualified;
        enum unqualified;
    }

    part def Vehicle {
        attribute maxAllowedMass : Real = 2500.0;
        attribute actualMass : Real;
        attribute status : QualificationStatus;
    }

    part def WeighingEquipment {
        attribute measuredMass : Real;
    }

    part def MassVerificationUnit {
        attribute threshold : Real = 2500.0;
    }

    part system {
        part vehicle : Vehicle;
        part scale : WeighingEquipment;
        part verifier : MassVerificationUnit;

        connect scale to verifier;
    }
}