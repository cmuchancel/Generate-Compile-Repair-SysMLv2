package AutomotiveConfigurationSpace {
    private import ScalarValues::*;

    enum def EngineType {
        enum FourCylinder;
        enum SixCylinder;
    }

    enum def TransmissionType {
        enum Manual;
        enum Automatic;
    }

    enum def CylinderDiameter {
        enum Diameter70mm;
        enum Diameter100mm;
    }

    part def Engine {
        attribute type : EngineType;
        attribute cylinderCount : Integer;
        attribute cylinderDiameter : CylinderDiameter;
    }

    part def Transmission {
        attribute type : TransmissionType;
    }

    part def Vehicle {
        part engine : Engine;
        part transmission : Transmission;
    }

    part fourCylEngine : Engine {
        attribute type : EngineType = EngineType::FourCylinder;
        attribute cylinderCount : Integer = 4;
        attribute cylinderDiameter : CylinderDiameter;
    }

    part sixCylEngine : Engine {
        attribute type : EngineType = EngineType::SixCylinder;
        attribute cylinderCount : Integer = 6;
        attribute cylinderDiameter : CylinderDiameter;
    }

    part manualTransmission : Transmission {
        attribute type : TransmissionType = TransmissionType::Manual;
    }

    part automaticTransmission : Transmission {
        attribute type : TransmissionType = TransmissionType::Automatic;
    }

    part valid4CylVehicle : Vehicle {
        part engine = fourCylEngine;
        part transmission = manualTransmission;
    }

    part valid6CylVehicle : Vehicle {
        part engine = sixCylEngine;
        part transmission = automaticTransmission;
    }
}