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
            (vehicle.engine = FourCylinder and vehicle.transmission = Manual) or
            (vehicle.engine = SixCylinder and vehicle.transmission = Automatic)
        }
    }

    item def HighPerformanceEngineRule {
        part vehicle : Vehicle;
        constraint {
            (vehicle.isHighPerformance and vehicle.engine = SixCylinder) or
            (not vehicle.isHighPerformance and vehicle.engine = FourCylinder)
        }
    }
}