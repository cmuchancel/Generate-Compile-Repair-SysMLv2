package VehicleMassVerificationSystem {
    private import ScalarValues::*;

    enum def InspectionResult {
        enum Qualified;
        enum Unqualified;
    }

    item def MassData {
        attribute measuredMass : Real;
    }

    part def Vehicle {
        attribute actualMass : Real;
    }

    part def WeighingEquipment {
        attribute maxCapacity : Real = 5000.0;
        port dataOut : MassDataPort;
    }

    part def MassVerificationUnit {
        attribute massLimit : Real = 2500.0;
        attribute result : InspectionResult;
        port dataIn : MassDataPort;
    }

    port def MassDataPort {
        attribute massValue : Real;
    }

    part def VehicleMassVerificationSystem {
        attribute systemName : String = "VehicleMassVerificationSystem";

        part vehicleUnderTest : Vehicle {
            attribute actualMass : Real;
        }

        part weighingEquipment : WeighingEquipment {
            attribute maxCapacity : Real = 5000.0;
            port dataOut : MassDataPort;
        }

        part massVerificationUnit : MassVerificationUnit {
            attribute massLimit : Real = 2500.0;
            attribute result : InspectionResult;
            port dataIn : MassDataPort;
        }

        connect weighingEquipment.dataOut to massVerificationUnit.dataIn;
    }
}