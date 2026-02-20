package VehicleMassCompliance {
    private import ScalarValues::*;

    enum def ComplianceStatus {
        enum COMPLIANT;
        enum NON_COMPLIANT;
    }

    part def Vehicle {
        attribute actualMass : Real;
        attribute maxPermissibleMass : Real = 2500.0;
        attribute complianceStatus : ComplianceStatus;
    }

    part def WeighingEquipment {
        attribute measuredMass : Real;
    }

    part def TestOperator {
    }

    part def ComplianceSystem {
        part vehicle : Vehicle;
        part weighingEquipment : WeighingEquipment;
        part testOperator : TestOperator;

        attribute isCompliant : Boolean;

        port def MassInputPort;
        port def StatusOutputPort;

        port massInput : MassInputPort;
        port statusOutput : StatusOutputPort;
    }

    item def MassData {
        attribute value : Real;
    }

    item def ComplianceResult {
        attribute status : ComplianceStatus;
    }

    part complianceSystem : ComplianceSystem;

    connect complianceSystem.weighingEquipment to complianceSystem.massInput;
    connect complianceSystem.statusOutput to complianceSystem.vehicle;
}