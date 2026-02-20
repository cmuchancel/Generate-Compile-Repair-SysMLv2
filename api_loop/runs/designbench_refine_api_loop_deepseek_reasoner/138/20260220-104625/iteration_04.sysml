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
        
        attribute engineType : EngineType = engine.type;
        attribute transmissionType : TransmissionType = transmission.type;
        
        constraint {
            if (isHighPerformance) then (engineType == SixCylinder) else (engineType == FourCylinder);
            if (engineType == FourCylinder) then (transmissionType == Manual) else (transmissionType == Automatic);
        }
    }
}