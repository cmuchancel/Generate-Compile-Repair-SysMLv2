package VehicleProductLine {
    private import ScalarValues::*;

    enum def PerformanceLevel {
        enum HIGH;
        enum STANDARD;
    }

    enum def EngineType {
        enum SIX_CYLINDER;
        enum FOUR_CYLINDER;
    }

    enum def TransmissionType {
        enum AUTOMATIC;
        enum MANUAL;
    }

    enum def WheelType {
        enum WIDE_RIM;
        enum NARROW_RIM;
    }

    part def Engine {
        attribute engineType : EngineType;
        attribute cylinders : Integer;
        attribute horsepower : Real;
    }

    part def Transmission {
        attribute transmissionType : TransmissionType;
        attribute gearCount : Integer;
    }

    part def Wheel {
        attribute wheelType : WheelType;
        attribute rimWidth : Real;
    }

    part def VehicleConfiguration {
        attribute performanceLevel : PerformanceLevel;
        attribute configName : String;
        part engine : Engine;
        part transmission : Transmission;
        part wheel : Wheel;
    }

    part def HighPerformanceEngine :> Engine {
        attribute engineType : EngineType = EngineType::SIX_CYLINDER;
        attribute cylinders : Integer = 6;
        attribute horsepower : Real = 400.0;
    }

    part def StandardEngine :> Engine {
        attribute engineType : EngineType = EngineType::FOUR_CYLINDER;
        attribute cylinders : Integer = 4;
        attribute horsepower : Real = 150.0;
    }

    part def AutomaticTransmission :> Transmission {
        attribute transmissionType : TransmissionType = TransmissionType::AUTOMATIC;
        attribute gearCount : Integer = 8;
    }

    part def ManualTransmission :> Transmission {
        attribute transmissionType : TransmissionType = TransmissionType::MANUAL;
        attribute gearCount : Integer = 6;
    }

    part def WideRimWheel :> Wheel {
        attribute wheelType : WheelType = WheelType::WIDE_RIM;
        attribute rimWidth : Real = 10.0;
    }

    part def NarrowRimWheel :> Wheel {
        attribute wheelType : WheelType = WheelType::NARROW_RIM;
        attribute rimWidth : Real = 7.0;
    }

    part def HighPerformanceConfiguration :> VehicleConfiguration {
        attribute performanceLevel : PerformanceLevel = PerformanceLevel::HIGH;
        attribute configName : String = "HighPerformance";
        part engine : HighPerformanceEngine;
        part transmission : AutomaticTransmission;
        part wheel : WideRimWheel;
    }

    part def StandardConfiguration :> VehicleConfiguration {
        attribute performanceLevel : PerformanceLevel = PerformanceLevel::STANDARD;
        attribute configName : String = "Standard";
        part engine : StandardEngine;
        part transmission : ManualTransmission;
        part wheel : NarrowRimWheel;
    }

    port def ConfigurationPort {
        attribute performanceLevel : PerformanceLevel;
    }

    part def VehicleProductLineSystem {
        attribute systemName : String = "VehicleProductLineSystem";
        part highPerfConfig : HighPerformanceConfiguration;
        part standardConfig : StandardConfiguration;
        port highPerfPort : ConfigurationPort;
        port standardPort : ConfigurationPort;
    }

    part def ProductLineManager {
        attribute managerName : String = "ProductLineManager";
        port managerPort : ConfigurationPort;
        part system : VehicleProductLineSystem;
    }

    part vehicleSystem : VehicleProductLineSystem;
    part manager : ProductLineManager;

    connect manager.managerPort to vehicleSystem.highPerfPort;
    connect manager.managerPort to vehicleSystem.standardPort;
}