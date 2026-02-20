package VehicleProductLineConfiguration {
    private import ScalarValues::*;

    enum def PerformanceRequirement {
        enum HighPerformance;
        enum StandardPerformance;
    }

    enum def EngineType {
        enum FourCylinder;
        enum SixCylinder;
    }

    enum def TransmissionType {
        enum Manual;
        enum Automatic;
    }

    enum def WheelType {
        enum NarrowRim;
        enum WideRim;
    }

    part def Engine {
        attribute engineType : EngineType;
    }

    part def Transmission {
        attribute transmissionType : TransmissionType;
    }

    part def Wheel {
        attribute wheelType : WheelType;
        attribute quantity : Integer = 4;
    }

    part def Vehicle {
        attribute performanceRequirement : PerformanceRequirement;
        part engine : Engine;
        part transmission : Transmission;
        part wheels : Wheel[4];
    }

    part def ConfigurationSystem {
        part vehicle : Vehicle;

        attribute selectedPerformance : PerformanceRequirement;

        port def InputPort {
            attribute performance : PerformanceRequirement;
        }

        port def OutputPort {
            attribute configuredVehicle : Vehicle;
        }

        port input : InputPort;
        port output : OutputPort;
    }

    part configurationSystem : ConfigurationSystem;

    connect configurationSystem.input to configurationSystem.vehicle {
        attribute performanceRequirement = configurationSystem.input.performance;
    }

    part highPerformanceEngine : Engine {
        attribute engineType = EngineType::SixCylinder;
    }

    part highPerformanceTransmission : Transmission {
        attribute transmissionType = TransmissionType::Automatic;
    }

    part highPerformanceWheels : Wheel {
        attribute wheelType = WheelType::WideRim;
    }

    part standardPerformanceEngine : Engine {
        attribute engineType = EngineType::FourCylinder;
    }

    part standardPerformanceTransmission : Transmission {
        attribute transmissionType = TransmissionType::Manual;
    }

    part standardPerformanceWheels : Wheel {
        attribute wheelType = WheelType::NarrowRim;
    }
}