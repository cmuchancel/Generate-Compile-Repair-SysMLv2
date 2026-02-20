package MassAggregation {
  private import ScalarValues::*;
  part def ObjectWithMass {
    attribute simpleMass : Real;
    attribute totalMass : Real;
  }
  part def simpleThing specializes ObjectWithMass {
    attribute totalMass : Real = simpleMass;
  }
  part def compositeThing specializes ObjectWithMass {
    part subcomponents : ObjectWithMass [*];
    attribute totalMass : Real = simpleMass + sum(subcomponents.totalMass);
  }
}