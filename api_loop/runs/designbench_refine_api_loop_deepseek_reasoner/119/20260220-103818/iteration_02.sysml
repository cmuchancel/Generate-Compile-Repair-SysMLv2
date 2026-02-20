package VehicleMassComplianceSystem {
    private import ScalarValues::*;

    enum def ComplianceStatus {
        enum Compliant,
        enum NonCompliant
    }

    item def MassData {
        attribute value : Real;
    }

    port def MassMeasurementPort {
        attribute measurement : MassData;
    }

    port def ComplianceResultPort {
        attribute status : ComplianceStatus;
    }

    part def Vehicle {
        attribute id : String;
        attribute measuredMass : Real;
        attribute maxPermissibleMass : Real = 2500.0;
        attribute complianceStatus : ComplianceStatus;
    }

    part def WeighingEquipment {
        port output : MassMeasurementPort;
        attribute currentMeasurement : MassData;
    }

    part def TestOperator {
        port control : MassMeasurementPort;
    }

    part def ComplianceVerifier {
        port input : MassMeasurementPort;
        port output : ComplianceResultPort;
        attribute limit : Real = 2500.0;
    }

    part def VehicleTestStation {
        part operator : TestOperator;
        part scale : WeighingEquipment;
        part verifier : ComplianceVerifier;
        part vehicle : Vehicle;

        connect operator.control to scale.output;
        connect scale.output to verifier.input;
        attribute testCount : Integer;
    }
}