package VehicleConfiguration {
    private import ScalarValues::*;

    enum def EngineType {
        fourCylinder,
        sixCylinder
    }

    enum def CylinderDiameter {
        small,
        large
    }

    enum def TransmissionType {
        manual,
        automatic
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