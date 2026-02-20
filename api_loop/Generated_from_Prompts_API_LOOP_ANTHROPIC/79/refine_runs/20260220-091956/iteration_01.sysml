package VehicleProductLine {
    private import ScalarValues::*;

    enum def CylinderDiameter {
        enum small;
        enum large;
    }

    enum def EngineType {
        enum fourCylinder;
        enum sixCylinder;
    }

    enum def TransmissionType {
        enum manual;
        enum automatic;
    }

    part def Cylinder {
        attribute diameter : CylinderDiameter;
    }

    part def FourCylinderEngine {
        attribute engineType : EngineType = EngineType::fourCylinder;
        attribute cylinderCount : Integer = 4;
        part cyl1 : Cylinder;
        part cyl2 : Cylinder;
        part cyl3 : Cylinder;
        part cyl4 : Cylinder;
    }

    part def SixCylinderEngine {
        attribute engineType : EngineType = EngineType::sixCylinder;
        attribute cylinderCount : Integer = 6;
        part cyl1 : Cylinder;
        part cyl2 : Cylinder;
        part cyl3 : Cylinder;
        part cyl4 : Cylinder;
        part cyl5 : Cylinder;
        part cyl6 : Cylinder;
    }

    part def ManualTransmission {
        attribute transmissionType : TransmissionType = TransmissionType::manual;
    }

    part def AutomaticTransmission {
        attribute transmissionType : TransmissionType = TransmissionType::automatic;
    }

    part def VehicleWithFourCylManual {
        part engine : FourCylinderEngine;
        part transmission : ManualTransmission;
    }

    part def VehicleWithFourCylAutomatic {
        part engine : FourCylinderEngine;
        part transmission : AutomaticTransmission;
    }

    part def VehicleWithSixCylAutomatic {
        part engine : SixCylinderEngine;
        part transmission : AutomaticTransmission;
    }

    part vehicleVariant1 : VehicleWithFourCylManual {
        part redefines engine {
            part redefines cyl1 { attribute redefines diameter = CylinderDiameter::small; }
            part redefines cyl2 { attribute redefines diameter = CylinderDiameter::small; }
            part redefines cyl3 { attribute redefines diameter = CylinderDiameter::small; }
            part redefines cyl4 { attribute redefines diameter = CylinderDiameter::small; }
        }
    }

    part vehicleVariant2 : VehicleWithFourCylAutomatic {
        part redefines engine {
            part redefines cyl1 { attribute redefines diameter = CylinderDiameter::large; }
            part redefines cyl2 { attribute redefines diameter = CylinderDiameter::large; }
            part redefines cyl3 { attribute redefines diameter = CylinderDiameter::large; }
            part redefines cyl4 { attribute redefines diameter = CylinderDiameter::large; }
        }
    }

    part vehicleVariant3 : VehicleWithSixCylAutomatic {
        part redefines engine {
            part redefines cyl1 { attribute redefines diameter = CylinderDiameter::small; }
            part redefines cyl2 { attribute redefines diameter = CylinderDiameter::small; }
            part redefines cyl3 { attribute redefines diameter = CylinderDiameter::small; }
            part redefines cyl4 { attribute redefines diameter = CylinderDiameter::small; }
            part redefines cyl5 { attribute redefines diameter = CylinderDiameter::small; }
            part redefines cyl6 { attribute redefines diameter = CylinderDiameter::small; }
        }
    }

    part vehicleVariant4 : VehicleWithSixCylAutomatic {
        part redefines engine {
            part redefines cyl1 { attribute redefines diameter = CylinderDiameter::large; }
            part redefines cyl2 { attribute redefines diameter = CylinderDiameter::large; }
            part redefines cyl3 { attribute redefines diameter = CylinderDiameter::large; }
            part redefines cyl4 { attribute redefines diameter = CylinderDiameter::large; }
            part redefines cyl5 { attribute redefines diameter = CylinderDiameter::large; }
            part redefines cyl6 { attribute redefines diameter = CylinderDiameter::large; }
        }
    }
}