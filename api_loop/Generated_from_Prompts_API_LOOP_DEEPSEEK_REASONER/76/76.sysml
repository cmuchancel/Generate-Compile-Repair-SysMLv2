package MassVerificationSystem {
    private import ScalarValues::*;

    item def MassValue {
        attribute value : Real;
    }

    port def MassDataPort;

    enum def QualificationResult {
        enum qualified;
        enum unqualified;
    }

    part def Vehicle {
        attribute mass : Real;
        attribute result : QualificationResult;
        port massPort : MassDataPort;
    }

    part def WeighingEquipment {
        attribute measuredMass : Real;
        port inputPort : MassDataPort;
        port outputPort : MassDataPort;
    }

    part def MassVerificationUnit {
        attribute threshold : Real = 2500.0;
        attribute verificationResult : QualificationResult;
        port inputPort : MassDataPort;
    }

    part def InspectionSystem {
        part vehicle : Vehicle;
        part weigher : WeighingEquipment;
        part verifier : MassVerificationUnit;

        connect vehicle.massPort to weigher.inputPort;
        connect weigher.outputPort to verifier.inputPort;
    }
}