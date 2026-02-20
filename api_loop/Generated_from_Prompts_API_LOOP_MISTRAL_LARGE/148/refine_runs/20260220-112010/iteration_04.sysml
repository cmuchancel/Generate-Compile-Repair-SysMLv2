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
        part composition : Composition[*];
    }

    part def Steel_980 : Alloy {
        attribute tensileStrengthName : String = "Tensile Strength";
        attribute tensileStrengthValue : Real = 980.0;
        attribute tensileStrengthUnit : String = "N/mm²";

        part iron : Composition;
        part carbon : Composition;
        part manganese : Composition;
    }

    part materialsDatabase {
        part steel980 : Steel_980;
        part steel980_iron : Composition {
            constituent = "Iron";
            massFraction = 0.97;
        }
        part steel980_carbon : Composition {
            constituent = "Carbon";
            massFraction = 0.002;
        }
        part steel980_manganese : Composition {
            constituent = "Manganese";
            massFraction = 0.015;
        }
        connect steel980.composition to steel980_iron;
        connect steel980.composition to steel980_carbon;
        connect steel980.composition to steel980_manganese;
    }
}