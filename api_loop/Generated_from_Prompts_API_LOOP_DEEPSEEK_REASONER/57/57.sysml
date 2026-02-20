package MassAggregationSystem {
    private import ScalarValues::*;
    
    part def Component {
        attribute basicMass : Real = 0.0;
        attribute minMassThreshold : Real = 0.0;
        
        part subcomponent : Component[*];
    }
}