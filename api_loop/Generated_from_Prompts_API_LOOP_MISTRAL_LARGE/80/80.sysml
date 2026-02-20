package ConfigurableEngine {
    private import ScalarValues::*;

    enum def CylinderDiameter {
        enum D70;
        enum D100;
    }

    part def Cylinder {
        attribute diameter : CylinderDiameter;
    }

    part def Engine {
        attribute cylinderCount : Integer;
        part cylinders : Cylinder[*];
    }

    part fourCylinderEngine : Engine {
        attribute cylinderCount : Integer = 4;
        part cylinders : Cylinder[4] {
            attribute :>> diameter = CylinderDiameter::D70;
        }
    }

    part sixCylinderEngine : Engine {
        attribute cylinderCount : Integer = 6;
        part cylinders : Cylinder[6] {
            attribute :>> diameter = CylinderDiameter::D70;
        }
    }

    part def EngineVariant {
        part engine : Engine;
    }

    part fourCylD70 : EngineVariant {
        part engine : fourCylinderEngine;
    }

    part fourCylD100 : EngineVariant {
        part engine : fourCylinderEngine {
            part cylinders : Cylinder[4] {
                attribute :>> diameter = CylinderDiameter::D100;
            }
        }
    }

    part sixCylD70 : EngineVariant {
        part engine : sixCylinderEngine;
    }

    part sixCylD100 : EngineVariant {
        part engine : sixCylinderEngine {
            part cylinders : Cylinder[6] {
                attribute :>> diameter = CylinderDiameter::D100;
            }
        }
    }
}