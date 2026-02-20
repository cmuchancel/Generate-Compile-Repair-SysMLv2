package MassManagement {
    private import ScalarValues::*;

    part def MassObject {
        attribute basicMass : Real;
        attribute subcomponents : MassObject[*];

        attribute totalMass : Real;
    }

    part def MassAggregator {
        part objects : MassObject[*];

        attribute aggregateMass : Real;
    }
}