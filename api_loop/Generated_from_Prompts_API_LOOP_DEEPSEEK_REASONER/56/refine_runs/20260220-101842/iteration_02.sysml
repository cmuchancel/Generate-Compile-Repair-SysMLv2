package MassAggregation {
    private import ScalarValues::*;

    part def Thing {
        attribute simpleMass : Real = 0;
        attribute totalMass : Real = 0;
    }

    part def SimpleThing :> Thing {
        attribute totalMass : Real = simpleMass;
    }

    part def CompositeThing :> Thing {
        part subcomponent : Thing[0..*];
    }
}