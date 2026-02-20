package EngineeringMaterialsDatabase {
    private import ScalarValues::*;

    part def Material {
        attribute atomicMass : Real;
        attribute tensileStrength : Real;
    }

    part def Metal {
        attribute atomicNumber : Integer;
    }

    part def Alloy {
        part constituents : Constituent[1..*];
        attribute tensileStrength : Real;
    }

    part def Constituent {
        attribute massFraction : Real;
        part material : Material;
    }

    part Iron : Metal {
        attribute atomicMass = 55.845;
        attribute atomicNumber = 26;
        attribute tensileStrength = 0.0;
    }

    part Carbon : Material {
        attribute atomicMass = 12.011;
        attribute tensileStrength = 0.0;
    }

    part Manganese : Metal {
        attribute atomicMass = 54.938;
        attribute atomicNumber = 25;
        attribute tensileStrength = 0.0;
    }

    part Steel_980 : Alloy {
        part ironConstituent : Constituent {
            attribute massFraction = 0.98;
            part material :> Iron;
        }
        part carbonConstituent : Constituent {
            attribute massFraction = 0.01;
            part material :> Carbon;
        }
        part manganeseConstituent : Constituent {
            attribute massFraction = 0.01;
            part material :> Manganese;
        }
        attribute tensileStrength = 980.0;
    }
}