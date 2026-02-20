package MassAggregation {
    private import ScalarValues::*;

    part def MassObject {
        attribute simpleMass : Real;
        attribute totalMass : Real;
    }

    part def SimpleThing :> MassObject {
        attribute :>> simpleMass : Real;
        attribute :>> totalMass : Real = simpleMass;
    }

    part def CompositeThing :> MassObject {
        attribute :>> simpleMass : Real;
        attribute :>> totalMass : Real;
        part subcomponents : MassObject[0..*];
    }
}