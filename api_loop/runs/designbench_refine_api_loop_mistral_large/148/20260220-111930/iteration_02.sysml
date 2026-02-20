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
        attribute baseMetal : String;
        attribute composition : Composition[*];
    }

    part def Steel_980 : Alloy {
        attribute tensileStrength : MechanicalProperty = (
            propertyName = "Tensile Strength",
            value = 980.0,
            unit = "N/mm^2"
        );
        attribute iron : Composition = (
            constituent = "Iron",
            massFraction = 0.97
        );
        attribute carbon : Composition = (
            constituent = "Carbon",
            massFraction = 0.002
        );
        attribute manganese : Composition = (
            constituent = "Manganese",
            massFraction = 0.015
        );
    }

    part materialsDatabase {
        part steel980 : Steel_980;
    }
}