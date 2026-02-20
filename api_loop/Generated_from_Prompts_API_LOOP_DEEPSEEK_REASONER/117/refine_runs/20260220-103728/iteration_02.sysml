package VehicleProductLine {
    private import ScalarValues::*;

    enum def PerformanceRequirement {
        value Standard;
        value High;
    }

    enum def EngineType {
        value I4;
        value V6;
    }

    enum def TransmissionType {
        value Manual;
        value Automatic;
    }

    enum def WheelType {
        value NarrowRim;
        value WideRim;
    }

    part def Engine {
        attribute type : EngineType;
    }

    part def Transmission {
        attribute type : TransmissionType;
    }

    part def Wheel {
        attribute type : WheelType;
    }

    part def VehicleConfiguration {
        attribute requirement : PerformanceRequirement;

        part engine : Engine;
        part transmission : Transmission;
        part wheels : Wheel[4];

        port def performancePort;
        port performancePort : performancePort;
    }

    part def ProductLineSystem {
        part config : VehicleConfiguration;

        port def systemPort;
        port systemPort : systemPort;

        connect config.performancePort to systemPort;
    }
}