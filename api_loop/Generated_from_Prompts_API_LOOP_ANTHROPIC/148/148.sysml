package EngineeringMaterialsDatabase {
    private import ScalarValues::*;

    enum def MaterialCategory {
        enum Metal;
        enum Alloy;
        enum Polymer;
        enum Ceramic;
        enum Composite;
    }

    part def Material {
        attribute materialName : String;
        attribute atomicMass : Real;
        attribute category : MaterialCategory;
    }

    part def Composition {
        attribute constituentName : String;
        attribute massFraction : Real;
    }

    part def MechanicalProperties {
        attribute tensileStrengthValue : Real;
        attribute tensileStrengthUnit : String;
    }

    part def Metal :> Material {
        attribute density : Real;
        attribute meltingPoint : Real;
    }

    part def Alloy :> Metal {
        attribute baseMetal : String;
        attribute numberOfConstituents : Integer;
        part compositions : Composition[1..*];
        part mechanicalProps : MechanicalProperties;
    }

    part def SteelAlloy :> Alloy {
        attribute steelGrade : String;
    }

    part Steel_980 : SteelAlloy {
        attribute materialName : String = "Steel_980";
        attribute category : MaterialCategory = MaterialCategory::Alloy;
        attribute density : Real = 7.85;
        attribute baseMetal : String = "Iron";
        attribute steelGrade : String = "High-Strength";
        attribute numberOfConstituents : Integer = 3;

        part ironComposition : Composition {
            attribute constituentName : String = "Iron";
            attribute massFraction : Real = 0.97;
        }

        part carbonComposition : Composition {
            attribute constituentName : String = "Carbon";
            attribute massFraction : Real = 0.009;
        }

        part manganeseComposition : Composition {
            attribute constituentName : String = "Manganese";
            attribute massFraction : Real = 0.021;
        }

        part mechanicalProps : MechanicalProperties {
            attribute tensileStrengthValue : Real = 980.0;
            attribute tensileStrengthUnit : String = "N/mm2";
        }
    }
}