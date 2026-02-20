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
            if isHighPerformance then {
                engine.type = EngineType::SixCylinder;
            } else {
                engine.type = EngineType::FourCylinder;
            }
            if engine.type = EngineType::FourCylinder then {
                transmission.type = TransmissionType::Manual;
            } else {
                transmission.type = TransmissionType::Automatic;
            }
        }
    }
}