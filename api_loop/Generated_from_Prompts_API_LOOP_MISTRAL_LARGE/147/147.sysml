package MaterialsScience {

    private import ScalarValues::*;

    enum def MassFractionUnit {
        enum PERCENT;
        enum PPM;
        enum FRACTION;
    }

    part def Substance {
        attribute name : String;
        attribute atomicMass : Real;
        attribute atomicMassUnit : String = "u";
    }

    part def Metal {
        attribute base : Substance;
    }

    part def Alloy {
        attribute name : String;
        attribute tensileStrength : Real;
        attribute tensileStrengthUnit : String = "N/mm^2";

        part components : Component[1..*];
    }

    part def Component {
        attribute massFraction : Real;
        attribute massFractionUnit : MassFractionUnit = MassFractionUnit::PERCENT;

        part material : Metal;
    }

    part ironSubstance : Substance {
        attribute name : String = "Iron";
        attribute atomicMass : Real = 55.845;
    }

    part iron : Metal {
        part base = ironSubstance;
    }

    part carbonSubstance : Substance {
        attribute name : String = "Carbon";
        attribute atomicMass : Real = 12.011;
    }

    part carbon : Metal {
        part base = carbonSubstance;
    }

    part manganeseSubstance : Substance {
        attribute name : String = "Manganese";
        attribute atomicMass : Real = 54.938;
    }

    part manganese : Metal {
        part base = manganeseSubstance;
    }

    part steel980 : Alloy {
        attribute name : String = "Steel 980";
        attribute tensileStrength : Real = 980.0;

        part components : Component[2] {
            part component1 : Component {
                attribute massFraction : Real = 98.0;
                part material = iron;
            }
            part component2 : Component {
                attribute massFraction : Real = 2.0;
                part material = carbon;
            }
        }
    }
}