package MassManagement {
    private import ScalarValues::*;

    part def MassObject {
        attribute basicMass : Real;
        attribute subcomponents : MassObject[*];
        attribute totalMass(minThreshold : Real = 0.0) : Real {
            return basicMass + subcomponents->sum(m | if m.totalMass(minThreshold) >= minThreshold then m.totalMass(minThreshold) else 0.0 endif);
        }
    }

    part def MassAggregator {
        part objects : MassObject[*];

        attribute aggregateMass(minThreshold : Real = 0.0) : Real {
            return objects->sum(o | o.totalMass(minThreshold));
        }
    }
}