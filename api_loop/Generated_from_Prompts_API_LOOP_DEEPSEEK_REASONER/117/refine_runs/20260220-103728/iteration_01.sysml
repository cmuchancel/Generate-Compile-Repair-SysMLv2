package VehicleProductLine {
    
    private import ScalarValues::*;
    
    enum def PerformanceRequirement {
        enum Standard
        enum High
    }
    
    enum def EngineType {
        enum I4
        enum V6
    }
    
    enum def TransmissionType {
        enum Manual
        enum Automatic
    }
    
    enum def WheelType {
        enum NarrowRim
        enum WideRim
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