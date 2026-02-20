package EngineeringMaterialsDatabase {
    private import ScalarValues::*;

    enum def MaterialType {
        enum Metal;
        enum Alloy;
        enum Polymer;
        enum Ceramic;
        enum Composite;
    }

    part def Material {
        attribute name : String;
        attribute materialType : MaterialType;
        attribute atomicMass : Real;
        attribute density : Real;
    }

    part def MechanicalProperty {
        attribute tensileStrength : Real;
        attribute yieldStrength : Real;
        attribute elongation : Real;
    }

    part def Composition {
        attribute constituent : String;
        attribute massFraction : Real;
    }

    part def Metal {
        attribute baseElement : String;
    }

    part def Alloy {
        attribute components : Composition[0..*];
    }

    part def Steel_980 {
        attribute tensileStrength : Real = 980.0;
        attribute components : Composition[3] = (
            Composition(
                constituent = "Iron",
                massFraction = 0.97
            ),
            Composition(
                constituent = "Carbon",
                massFraction = 0.002
            ),
            Composition(
                constituent = "Manganese",
                massFraction = 0.02
            )
        );
    }

    part materialsDatabase : EngineeringMaterialsDatabase {
        part steel980 : Steel_980;
    }
}