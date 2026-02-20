package VehicleProductLine {
    private import ScalarValues::*;

    enum def PerformanceRequirement {
        standard;
        high;
    }

    enum def EngineType {
        I4;
        V6;
    }

    enum def TransmissionType {
        manual;
        automatic;
    }

    enum def WheelType {
        narrowRim;
        wideRim;
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
        port pPerformancePort : performancePort;
    }

    part def ProductLineSystem {
        part config : VehicleConfiguration;

        port def systemPort;
        port pSystemPort : systemPort;

        connect config.pPerformancePort to pSystemPort;
    }
}