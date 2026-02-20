package VehiclePowertrain {
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
    }

    part def PowertrainConfigurator {
        part vehicle : Vehicle;

        constraint highPerfEngine {
            vehicle.isHighPerformance implies vehicle.engine.type == EngineType::SixCylinder
        }

        constraint standardEngine {
            not vehicle.isHighPerformance implies vehicle.engine.type == EngineType::FourCylinder
        }

        constraint validCombination1 {
            vehicle.engine.type == EngineType::FourCylinder implies vehicle.transmission.type == TransmissionType::Manual
        }

        constraint validCombination2 {
            vehicle.engine.type == EngineType::SixCylinder implies vehicle.transmission.type == TransmissionType::Automatic
        }
    }
}