package MassAggregation {
    private import ScalarValues::*;

    part def simpleThing {
        attribute simpleMass : Real;
        attribute totalMass : Real = simpleMass;
    }

    part def compositeThing {
        attribute simpleMass : Real;
        attribute totalMass : Real;
        part subcomponents : simpleThing[0..*];

        constraint totalMassConstraint {
            totalMass == simpleMass + subcomponents.totalMass->sum()
        }
    }
}