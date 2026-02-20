package PowertrainConfiguration {
    private import ScalarValues::*;

    enum def EngineType {
        FourCylinder,
        SixCylinder
    }

    enum def TransmissionType {
        Manual,
        Automatic
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
            (isHighPerformance = true) implies (engine.type = EngineType::SixCylinder);
            (isHighPerformance = false) implies (engine.type = EngineType::FourCylinder);
            
            (engine.type = EngineType::FourCylinder) implies (transmission.type = TransmissionType::Manual);
            (engine.type = EngineType::SixCylinder) implies (transmission.type = TransmissionType::Automatic);
        }
    }
}