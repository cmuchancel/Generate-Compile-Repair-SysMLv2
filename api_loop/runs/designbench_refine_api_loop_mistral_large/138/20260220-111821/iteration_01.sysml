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

        constraint {
            if vehicle.isHighPerformance then
                vehicle.engine.type == EngineType::SixCylinder
            else
                vehicle.engine.type == EngineType::FourCylinder
            end;

            (vehicle.engine.type == EngineType::FourCylinder and vehicle.transmission.type == TransmissionType::Manual) or
            (vehicle.engine.type == EngineType::SixCylinder and vehicle.transmission.type == TransmissionType::Automatic)
        }
    }
}