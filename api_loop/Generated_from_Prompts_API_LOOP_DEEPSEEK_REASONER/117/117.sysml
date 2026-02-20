package VehicleProductLine {
    private import ScalarValues::*;

    enum def EngineType {
        enum sixCylinder;
        enum fourCylinder;
    }

    enum def TransmissionType {
        enum automatic;
        enum manual;
    }

    enum def WheelType {
        enum wideRim;
        enum narrowRim;
    }

    part def Engine {
        attribute type : EngineType;
    }

    part def Transmission {
        attribute type : TransmissionType;
    }

    part def Wheels {
        attribute type : WheelType;
    }

    part def Vehicle {
        part engine : Engine;
        part transmission : Transmission;
        part wheels : Wheels;
    }

    part def HighPerformanceVehicle :> Vehicle {
        part engine : Engine {
            attribute type : EngineType = EngineType::sixCylinder;
        }
        part transmission : Transmission {
            attribute type : TransmissionType = TransmissionType::automatic;
        }
        part wheels : Wheels {
            attribute type : WheelType = WheelType::wideRim;
        }
    }

    part def StandardPerformanceVehicle :> Vehicle {
        part engine : Engine {
            attribute type : EngineType = EngineType::fourCylinder;
        }
        part transmission : Transmission {
            attribute type : TransmissionType = TransmissionType::manual;
        }
        part wheels : Wheels {
            attribute type : WheelType = WheelType::narrowRim;
        }
    }
}