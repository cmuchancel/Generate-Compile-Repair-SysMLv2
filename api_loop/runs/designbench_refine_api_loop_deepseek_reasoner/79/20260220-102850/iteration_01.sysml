package VehicleConfiguration {
    private import ScalarValues::*;

    enum def EngineType {
        enum fourCylinder,
        enum sixCylinder
    }

    enum def CylinderDiameter {
        enum small,
        enum large
    }

    enum def TransmissionType {
        enum manual,
        enum automatic
    }

    part def Cylinder {
        attribute diameter : CylinderDiameter;
    }

    part def Engine {
        attribute type : EngineType;
        part cylinders : Cylinder[0..*];
    }

    part def Transmission {
        attribute type : TransmissionType;
    }

    part def Vehicle {
        part engine : Engine;
        part transmission : Transmission;
    }
}