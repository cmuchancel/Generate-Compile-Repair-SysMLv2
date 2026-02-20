package VehicleMassCompliance {
    private import ScalarValues::*;

    enum def ComplianceStatus {
        enum Compliant;
        enum NonCompliant;
    }

    item def Vehicle {
        attribute vehicleId : String;
        attribute actualMass : Real;
    }

    item def MassReading {
        attribute measuredMass : Real;
        attribute vehicleId : String;
    }

    item def ComplianceResult {
        attribute vehicleId : String;
        attribute measuredMass : Real;
        attribute maximumPermissibleMass : Real;
        attribute status : ComplianceStatus;
    }

    part def WeighingEquipment {
        attribute calibrated : Boolean;
        port measurementPort : MeasurementPort;
    }

    part def TestOperator {
        attribute operatorId : String;
        port operatorOutputPort : OperatorPort;
    }

    part def ComplianceChecker {
        attribute maximumPermissibleMass : Real = 2500.0;
        port checkerInputPort : CheckerInputPort;
        port checkerOutputPort : CheckerOutputPort;
    }

    part def ComplianceRecord {
        attribute totalVehiclesTested : Integer;
        attribute compliantCount : Integer;
        attribute nonCompliantCount : Integer;
    }

    part def VehicleMassComplianceSystem {
        attribute systemId : String;
        attribute maximumPermissibleMass : Real = 2500.0;

        part weighingEquipment : WeighingEquipment {
            attribute calibrated : Boolean = true;
        }

        part testOperator : TestOperator {
            attribute operatorId : String = "OP-001";
        }

        part complianceChecker : ComplianceChecker {
            attribute maximumPermissibleMass : Real = 2500.0;
        }

        part complianceRecord : ComplianceRecord {
            attribute totalVehiclesTested : Integer = 0;
            attribute compliantCount : Integer = 0;
            attribute nonCompliantCount : Integer = 0;
        }

        part vehicle1 : Vehicle {
            attribute vehicleId : String = "V-001";
            attribute actualMass : Real = 2200.0;
        }

        part vehicle2 : Vehicle {
            attribute vehicleId : String = "V-002";
            attribute actualMass : Real = 2500.0;
        }

        part vehicle3 : Vehicle {
            attribute vehicleId : String = "V-003";
            attribute actualMass : Real = 2750.0;
        }

        part result1 : ComplianceResult {
            attribute vehicleId : String = "V-001";
            attribute measuredMass : Real = 2200.0;
            attribute maximumPermissibleMass : Real = 2500.0;
            attribute status : ComplianceStatus = ComplianceStatus::Compliant;
        }

        part result2 : ComplianceResult {
            attribute vehicleId : String = "V-002";
            attribute measuredMass : Real = 2500.0;
            attribute maximumPermissibleMass : Real = 2500.0;
            attribute status : ComplianceStatus = ComplianceStatus::Compliant;
        }

        part result3 : ComplianceResult {
            attribute vehicleId : String = "V-003";
            attribute measuredMass : Real = 2750.0;
            attribute maximumPermissibleMass : Real = 2500.0;
            attribute status : ComplianceStatus = ComplianceStatus::NonCompliant;
        }

        connect testOperator.operatorOutputPort to weighingEquipment.measurementPort;
        connect weighingEquipment.measurementPort to complianceChecker.checkerInputPort;
        connect complianceChecker.checkerOutputPort to complianceRecord;
    }

    port def MeasurementPort {
        attribute massValue : Real;
    }

    port def OperatorPort {
        attribute operatorAction : String;
    }

    port def CheckerInputPort {
        attribute incomingMass : Real;
    }

    port def CheckerOutputPort {
        attribute complianceDecision : ComplianceStatus;
    }
}