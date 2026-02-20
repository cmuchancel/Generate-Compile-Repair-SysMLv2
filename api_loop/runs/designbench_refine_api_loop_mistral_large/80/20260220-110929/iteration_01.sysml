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
        part cylinders : Cylinder[4];
    }

    part sixCylinderEngine : Engine {
        attribute cylinderCount : Integer = 6;
        part cylinders : Cylinder[6];
    }

    part def EngineVariant {
        part engine : Engine;
    }

    part fourCylD70 : EngineVariant {
        part engine : fourCylinderEngine {
            part cylinders : Cylinder[4] {
                cylinders[1].diameter = CylinderDiameter::D70;
                cylinders[2].diameter = CylinderDiameter::D70;
                cylinders[3].diameter = CylinderDiameter::D70;
                cylinders[4].diameter = CylinderDiameter::D70;
            }
        }
    }

    part fourCylD100 : EngineVariant {
        part engine : fourCylinderEngine {
            part cylinders : Cylinder[4] {
                cylinders[1].diameter = CylinderDiameter::D100;
                cylinders[2].diameter = CylinderDiameter::D100;
                cylinders[3].diameter = CylinderDiameter::D100;
                cylinders[4].diameter = CylinderDiameter::D100;
            }
        }
    }

    part sixCylD70 : EngineVariant {
        part engine : sixCylinderEngine {
            part cylinders : Cylinder[6] {
                cylinders[1].diameter = CylinderDiameter::D70;
                cylinders[2].diameter = CylinderDiameter::D70;
                cylinders[3].diameter = CylinderDiameter::D70;
                cylinders[4].diameter = CylinderDiameter::D70;
                cylinders[5].diameter = CylinderDiameter::D70;
                cylinders[6].diameter = CylinderDiameter::D70;
            }
        }
    }

    part sixCylD100 : EngineVariant {
        part engine : sixCylinderEngine {
            part cylinders : Cylinder[6] {
                cylinders[1].diameter = CylinderDiameter::D100;
                cylinders[2].diameter = CylinderDiameter::D100;
                cylinders[3].diameter = CylinderDiameter::D100;
                cylinders[4].diameter = CylinderDiameter::D100;
                cylinders[5].diameter = CylinderDiameter::D100;
                cylinders[6].diameter = CylinderDiameter::D100;
            }
        }
    }
}