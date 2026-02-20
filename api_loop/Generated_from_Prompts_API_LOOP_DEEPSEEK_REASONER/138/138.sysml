package PowertrainConfig {
    private import ScalarValues::*;

    enum def EngineType {
        enum FourCylinder;
        enum SixCylinder;
    }

    enum def TransmissionType {
        enum Manual;
        enum Automatic;
    }

    part def PowertrainCombination {
        attribute engine : EngineType;
        attribute transmission : TransmissionType;
    }

    part def Vehicle {
        attribute isHighPerformance : Boolean;
        part powertrain : PowertrainCombination;
    }

    part FourCylinderManual : PowertrainCombination {
        attribute engine : EngineType = EngineType::FourCylinder;
        attribute transmission : TransmissionType = TransmissionType::Manual;
    }

    part SixCylinderAutomatic : PowertrainCombination {
        attribute engine : EngineType = EngineType::SixCylinder;
        attribute transmission : TransmissionType = TransmissionType::Automatic;
    }
}