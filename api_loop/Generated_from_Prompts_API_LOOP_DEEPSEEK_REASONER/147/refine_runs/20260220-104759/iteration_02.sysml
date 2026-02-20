package AlloyModel {
    private import ScalarValues::*;

    part def Metal {
        attribute atomicMass : Real;
    }

    item def MassFraction {
        attribute fractionValue : Real;
        part component : Metal;
    }

    part def Alloy {
        attribute name : String;
        attribute tensileStrength : Real;
        part composition : MassFraction[2..*];
    }

    part def MaterialDatabase {
        part iron : Metal {
            atomicMass = 55.845;
        }
        part carbon : Metal {
            atomicMass = 12.011;
        }
        part manganese : Metal {
            atomicMass = 54.938;
        }
        part steel980 : Alloy {
            name = "Steel 980";
            tensileStrength = 980.0;
        }
        part comp1 : MassFraction {
            fractionValue = 0.98;
        }
        part comp2 : MassFraction {
            fractionValue = 0.02;
        }
        connect steel980.composition to comp1;
        connect steel980.composition to comp2;
        connect comp1.component to iron;
        connect comp2.component to carbon;
    }
}