package PowertrainConfiguration {
    private import ScalarValues::*;

    enum def EngineType {
        enum FourCylinder;
        enum SixCylinder;
    }

    enum def TransmissionType {
        enum Manual;
        enum Automatic;
    }

    part def Engine {
        attribute type : EngineType;
    }

    part def Transmission {
        attribute type : TransmissionType;
    }

    part def Vehicle {
        attribute isHighPerformance : Boolean;

        part engine : Engine;
        part transmission : Transmission;

        constraint {
            (isHighPerformance and engine.type = EngineType::SixCylinder) or
            (not isHighPerformance and engine.type = EngineType::FourCylinder);
            (engine.type = EngineType::FourCylinder and transmission.type = TransmissionType::Manual) or
            (engine.type = EngineType::SixCylinder and transmission.type = TransmissionType::Automatic);
        }
    }
}