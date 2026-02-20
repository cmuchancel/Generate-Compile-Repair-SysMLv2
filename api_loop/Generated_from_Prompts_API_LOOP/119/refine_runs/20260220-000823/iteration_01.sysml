package VehicleMassCompliance {
    private import ScalarValues::*;

    enum def ComplianceStatus {
        compliant;
        non_compliant;
    }

    part def Vehicle {
        attribute actualMass: Real;
        attribute status: ComplianceStatus;
    }

    part def WeighingEquipment {
        attribute measuredMass: Real;
    }

    part def TestOperator {
        attribute operatorId: String;
    }

    part def MassComplianceSystem {
        attribute maxPermissibleMass: Real = 2500.0;
        part equipment: WeighingEquipment;
        part operator: TestOperator;
        part vehicleUnderTest: Vehicle;
    }
}