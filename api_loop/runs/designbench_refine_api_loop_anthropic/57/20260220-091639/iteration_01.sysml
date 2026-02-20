package MassManagement {
    private import ScalarValues::*;

    attribute def BasicMass {
        attribute value : Real;
    }

    attribute def MinMassThreshold {
        attribute value : Real;
    }

    attribute def TotalMass {
        attribute value : Real;
    }

    part def Component {
        attribute basicMass : Real;
        attribute totalMass : Real;
        attribute minMassThreshold : Real;
    }

    part def Subcomponent {
        attribute basicMass : Real;
        attribute totalMass : Real;
    }

    part def CompositeObject {
        attribute basicMass : Real;
        attribute totalMass : Real;
        attribute minMassThreshold : Real;

        part subcomponent1 : Subcomponent;
        part subcomponent2 : Subcomponent;
        part subcomponent3 : Subcomponent;
    }

    part def FilteredCompositeObject {
        attribute basicMass : Real;
        attribute totalMass : Real;
        attribute minMassThreshold : Real;

        part subcomponent1 : Subcomponent;
        part subcomponent2 : Subcomponent;
        part subcomponent3 : Subcomponent;
    }

    part def MassAggregationSystem {
        attribute systemTotalMass : Real;
        attribute systemMinMassThreshold : Real;

        part composite1 : CompositeObject;
        part composite2 : FilteredCompositeObject;
        part standalone : Component;
    }

    part massSystem : MassAggregationSystem {
        attribute systemTotalMass : Real = 0.0;
        attribute systemMinMassThreshold : Real = 0.0;

        part composite1 : CompositeObject {
            attribute basicMass : Real = 10.0;
            attribute totalMass : Real = 40.0;
            attribute minMassThreshold : Real = 0.0;

            part subcomponent1 : Subcomponent {
                attribute basicMass : Real = 5.0;
                attribute totalMass : Real = 5.0;
            }
            part subcomponent2 : Subcomponent {
                attribute basicMass : Real = 15.0;
                attribute totalMass : Real = 15.0;
            }
            part subcomponent3 : Subcomponent {
                attribute basicMass : Real = 10.0;
                attribute totalMass : Real = 10.0;
            }
        }

        part composite2 : FilteredCompositeObject {
            attribute basicMass : Real = 8.0;
            attribute totalMass : Real = 33.0;
            attribute minMassThreshold : Real = 10.0;

            part subcomponent1 : Subcomponent {
                attribute basicMass : Real = 5.0;
                attribute totalMass : Real = 5.0;
            }
            part subcomponent2 : Subcomponent {
                attribute basicMass : Real = 15.0;
                attribute totalMass : Real = 15.0;
            }
            part subcomponent3 : Subcomponent {
                attribute basicMass : Real = 10.0;
                attribute totalMass : Real = 10.0;
            }
        }

        part standalone : Component {
            attribute basicMass : Real = 20.0;
            attribute totalMass : Real = 20.0;
            attribute minMassThreshold : Real = 0.0;
        }
    }
}