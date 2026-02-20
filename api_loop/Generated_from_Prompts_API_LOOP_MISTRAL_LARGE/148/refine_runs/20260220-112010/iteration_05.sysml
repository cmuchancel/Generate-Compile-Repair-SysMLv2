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

    part def Composition {
        attribute constituent : String;
        attribute massFraction : Real;
    }

    part def MechanicalProperty {
        attribute propertyName : String;
        attribute value : Real;
        attribute unit : String;
    }

    part def Metal {
        attribute baseElement : String;
    }

    part def Alloy {
        attribute baseMetal : Metal;
        part compositions : Composition[*];
    }

    part def Steel_980 : Alloy {
        attribute tensileStrengthName : String = "Tensile Strength";
        attribute tensileStrengthValue : Real = 980.0;
        attribute tensileStrengthUnit : String = "N/mm²";

        part iron : Composition {
            attribute constituent : String = "Iron";
            attribute massFraction : Real = 0.97;
        }
        part carbon : Composition {
            attribute constituent : String = "Carbon";
            attribute massFraction : Real = 0.002;
        }
        part manganese : Composition {
            attribute constituent : String = "Manganese";
            attribute massFraction : Real = 0.015;
        }
    }

    part materialsDatabase {
        part steel980 : Steel_980;
        connect steel980.compositions to steel980.iron;
        connect steel980.compositions to steel980.carbon;
        connect steel980.compositions to steel980.manganese;
    }
}