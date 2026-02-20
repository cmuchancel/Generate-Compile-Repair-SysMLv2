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

    part def Vehicle {
        attribute isHighPerformance : Boolean;
        attribute engine : EngineType;
        attribute transmission : TransmissionType;
    }

    item def ValidPowertrainCombination {
        part vehicle : Vehicle;
        constraint {
            (vehicle.engine = EngineType::FourCylinder and vehicle.transmission = TransmissionType::Manual) or
            (vehicle.engine = EngineType::SixCylinder and vehicle.transmission = TransmissionType::Automatic)
        }
    }

    item def HighPerformanceEngineRule {
        part vehicle : Vehicle;
        constraint {
            (vehicle.isHighPerformance and vehicle.engine = EngineType::SixCylinder) or
            (not vehicle.isHighPerformance and vehicle.engine = EngineType::FourCylinder)
        }
    }
}