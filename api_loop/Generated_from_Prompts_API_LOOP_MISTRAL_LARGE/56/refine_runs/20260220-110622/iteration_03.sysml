package MassAggregation {
    private import ScalarValues::*;

    part def simpleThing {
        attribute simpleMass : Real;
        attribute totalMass : Real = simpleMass;
    }

    part def compositeThing {
        attribute simpleMass : Real;
        attribute totalMass : Real = simpleMass;
        part subcomponents : simpleThing[0..*];

        attribute sumOfSubcomponentMasses : Real = 0.0;

        for sub in subcomponents {
            sumOfSubcomponentMasses += sub.totalMass;
        }

        totalMass = simpleMass + sumOfSubcomponentMasses;
    }
}