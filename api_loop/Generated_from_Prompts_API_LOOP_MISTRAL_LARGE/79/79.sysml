package VehicleProductLine {
    private import ScalarValues::*;

    enum def EngineType {
        enum FourCylinder;
        enum SixCylinder;
    }

    enum def CylinderDiameter {
        enum Small = 70;
        enum Large = 100;
    }

    enum def TransmissionType {
        enum Manual;
        enum Automatic;
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

        attribute isValid : Boolean = (
            (engine.type == EngineType::FourCylinder and transmission.type == TransmissionType::Manual) or
            (engine.type == EngineType::SixCylinder and transmission.type == TransmissionType::Automatic) or
            (engine.type == EngineType::FourCylinder and transmission.type == TransmissionType::Automatic)
        );
    }

    part fourCylEngine : Engine {
        type = EngineType::FourCylinder;
        cylinderCount = 4;
    }

    part sixCylEngine : Engine {
        type = EngineType::SixCylinder;
        cylinderCount = 6;
    }

    part manualTransmission : Transmission {
        type = TransmissionType::Manual;
    }

    part automaticTransmission : Transmission {
        type = TransmissionType::Automatic;
    }

    part validVehicle1 : Vehicle {
        engine = fourCylEngine;
        transmission = manualTransmission;
    }

    part validVehicle2 : Vehicle {
        engine = fourCylEngine;
        transmission = automaticTransmission;
    }

    part validVehicle3 : Vehicle {
        engine = sixCylEngine;
        transmission = automaticTransmission;
    }
}