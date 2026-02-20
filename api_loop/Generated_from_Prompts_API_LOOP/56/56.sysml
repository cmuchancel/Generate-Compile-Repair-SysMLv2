package MassAggregation {
  private import ScalarValues::*;
  part def ObjectWithMass {
    attribute simpleMass : Real;
    attribute subcomponentsTotalMass : Real = 0;
    attribute totalMass : Real = simpleMass + subcomponentsTotalMass;
  }
  part def simpleThing specializes ObjectWithMass {
  }
  part def compositeThing specializes ObjectWithMass {
    part subcomponents : ObjectWithMass [*];
  }
}