package MassAggregationSystem {
    private import ScalarValues::*;

    part def MassComponent {
        attribute basicMass : Real;
    }

    part def CompositeMassComponent {
        attribute basicMass : Real;
        part subcomponents[*] : MassComponent;
        attribute totalMass : Real;
        attribute minMassThreshold : Real;
        attribute totalMassFiltered : Real;
    }

    part def MassAggregationManager {
        part components[*] : MassComponent;
    }
}