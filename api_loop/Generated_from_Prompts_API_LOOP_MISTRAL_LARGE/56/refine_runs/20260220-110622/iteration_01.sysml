package MassAggregation {
    private import ScalarValues::*;

    part def simpleThing {
        attribute simpleMass : Real;
        attribute totalMass : Real = simpleMass;
    }

    part def compositeThing {
        attribute simpleMass : Real;
        attribute totalMass : Real = simpleMass + subcomponents.totalMass->sum();
        part subcomponents : simpleThing[*] nonunique;
    }
}