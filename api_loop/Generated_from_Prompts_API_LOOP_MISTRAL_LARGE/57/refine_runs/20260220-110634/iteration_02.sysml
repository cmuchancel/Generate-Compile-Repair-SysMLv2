package MassManagement {
    private import ScalarValues::*;

    part def MassObject {
        attribute basicMass : Real;
        attribute subcomponents : MassObject[*];

        attribute totalMass(minThreshold : Real = 0.0) : Real;
    }

    part def MassAggregator {
        part objects : MassObject[*];

        attribute aggregateMass(minThreshold : Real = 0.0) : Real;
    }
}