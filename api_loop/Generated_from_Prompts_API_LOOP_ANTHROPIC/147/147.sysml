package AlloyManagement {
    private import ScalarValues::*;

    enum def PhysicalState {
        enum Solid;
        enum Liquid;
        enum Gas;
    }

    part def Substance {
        attribute substanceName : String;
        attribute atomicMass : Real;
        attribute physicalState : PhysicalState;
    }

    part def Metal {
        attribute metalName : String;
        attribute atomicMass : Real;
        attribute physicalState : PhysicalState;
        attribute symbol : String;
    }

    part def MassComponent {
        attribute materialName : String;
        attribute massFraction : Real;
    }

    part def MechanicalProperties {
        attribute tensileStrength : Real;
        attribute tensileStrengthUnit : String;
    }

    part def Alloy {
        attribute alloyName : String;
        attribute tensileStrength : Real;
        attribute tensileStrengthUnit : String;
        part component1 : MassComponent;
        part component2 : MassComponent;
        part component3 : MassComponent;
        part mechanicalProps : MechanicalProperties;
    }

    part iron : Metal {
        attribute redefines metalName = "Iron";
        attribute redefines atomicMass = 55.845;
        attribute redefines symbol = "Fe";
    }

    part carbon : Metal {
        attribute redefines metalName = "Carbon";
        attribute redefines atomicMass = 12.011;
        attribute redefines symbol = "C";
    }

    part manganese : Metal {
        attribute redefines metalName = "Manganese";
        attribute redefines atomicMass = 54.938;
        attribute redefines symbol = "Mn";
    }

    part steel980 : Alloy {
        attribute redefines alloyName = "Steel 980";
        attribute redefines tensileStrength = 980.0;
        attribute redefines tensileStrengthUnit = "N/mm2";
        part redefines component1 : MassComponent {
            attribute redefines materialName = "Iron";
            attribute redefines massFraction = 0.972;
        }
        part redefines component2 : MassComponent {
            attribute redefines materialName = "Carbon";
            attribute redefines massFraction = 0.018;
        }
        part redefines component3 : MassComponent {
            attribute redefines materialName = "Manganese";
            attribute redefines massFraction = 0.010;
        }
        part redefines mechanicalProps : MechanicalProperties {
            attribute redefines tensileStrength = 980.0;
            attribute redefines tensileStrengthUnit = "N/mm2";
        }
    }
}