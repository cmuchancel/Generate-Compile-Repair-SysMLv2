package AutomotiveConfig {
    private import ScalarValues::*;

    enum def CylinderDiameter {
        enum d70;
        enum d100;
    }

    enum def TransmissionType {
        enum Manual;
        enum Automatic;
    }

    enum def EngineType {
        enum FourCylinder;
        enum SixCylinder;
    }

    part def Cylinder {
        attribute diameter : CylinderDiameter;
    }

    part def Engine {
        attribute engineType : EngineType;
    }

    part def FourCylinderEngine :> Engine {
        attribute redefines engineType = EngineType::FourCylinder;
        part cylinder1 : Cylinder;
        part cylinder2 : Cylinder;
        part cylinder3 : Cylinder;
        part cylinder4 : Cylinder;
    }

    part def SixCylinderEngine :> Engine {
        attribute redefines engineType = EngineType::SixCylinder;
        part cylinder1 : Cylinder;
        part cylinder2 : Cylinder;
        part cylinder3 : Cylinder;
        part cylinder4 : Cylinder;
        part cylinder5 : Cylinder;
        part cylinder6 : Cylinder;
    }

    part def Transmission {
        attribute transmissionType : TransmissionType;
    }

    part def ManualTransmission :> Transmission {
        attribute redefines transmissionType = TransmissionType::Manual;
    }

    part def AutomaticTransmission :> Transmission {
        attribute redefines transmissionType = TransmissionType::Automatic;
    }

    part def Vehicle {
        part engine : Engine[1];
        part transmission : Transmission[1];
    }

    part def VehicleWith4CylManual :> Vehicle {
        part redefines engine : FourCylinderEngine[1];
        part redefines transmission : ManualTransmission[1];
    }

    part def VehicleWith6CylAutomatic :> Vehicle {
        part redefines engine : SixCylinderEngine[1];
        part redefines transmission : AutomaticTransmission[1];
    }

    part vehicle4CylManual70 : VehicleWith4CylManual {
        part redefines engine : FourCylinderEngine {
            part redefines cylinder1 : Cylinder { attribute redefines diameter = CylinderDiameter::d70; }
            part redefines cylinder2 : Cylinder { attribute redefines diameter = CylinderDiameter::d70; }
            part redefines cylinder3 : Cylinder { attribute redefines diameter = CylinderDiameter::d70; }
            part redefines cylinder4 : Cylinder { attribute redefines diameter = CylinderDiameter::d70; }
        }
    }

    part vehicle4CylManual100 : VehicleWith4CylManual {
        part redefines engine : FourCylinderEngine {
            part redefines cylinder1 : Cylinder { attribute redefines diameter = CylinderDiameter::d100; }
            part redefines cylinder2 : Cylinder { attribute redefines diameter = CylinderDiameter::d100; }
            part redefines cylinder3 : Cylinder { attribute redefines diameter = CylinderDiameter::d100; }
            part redefines cylinder4 : Cylinder { attribute redefines diameter = CylinderDiameter::d100; }
        }
    }

    part vehicle6CylAuto70 : VehicleWith6CylAutomatic {
        part redefines engine : SixCylinderEngine {
            part redefines cylinder1 : Cylinder { attribute redefines diameter = CylinderDiameter::d70; }
            part redefines cylinder2 : Cylinder { attribute redefines diameter = CylinderDiameter::d70; }
            part redefines cylinder3 : Cylinder { attribute redefines diameter = CylinderDiameter::d70; }
            part redefines cylinder4 : Cylinder { attribute redefines diameter = CylinderDiameter::d70; }
            part redefines cylinder5 : Cylinder { attribute redefines diameter = CylinderDiameter::d70; }
            part redefines cylinder6 : Cylinder { attribute redefines diameter = CylinderDiameter::d70; }
        }
    }

    part vehicle6CylAuto100 : VehicleWith6CylAutomatic {
        part redefines engine : SixCylinderEngine {
            part redefines cylinder1 : Cylinder { attribute redefines diameter = CylinderDiameter::d100; }
            part redefines cylinder2 : Cylinder { attribute redefines diameter = CylinderDiameter::d100; }
            part redefines cylinder3 : Cylinder { attribute redefines diameter = CylinderDiameter::d100; }
            part redefines cylinder4 : Cylinder { attribute redefines diameter = CylinderDiameter::d100; }
            part redefines cylinder5 : Cylinder { attribute redefines diameter = CylinderDiameter::d100; }
            part redefines cylinder6 : Cylinder { attribute redefines diameter = CylinderDiameter::d100; }
        }
    }
}