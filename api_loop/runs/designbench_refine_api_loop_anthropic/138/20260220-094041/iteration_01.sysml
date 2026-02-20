package PowertrainConfiguration {
    private import ScalarValues::*;

    enum def EngineType {
        enum FourCylinder;
        enum SixCylinder;
    }

    enum def TransmissionType {
        enum Manual;
        enum Automatic;
    }

    enum def PerformanceModel {
        enum Standard;
        enum HighPerformance;
    }

    part def Engine {
        attribute engineType : EngineType;
    }

    part def Transmission {
        attribute transmissionType : TransmissionType;
    }

    part def FourCylinderEngine :> Engine {
        attribute redefines engineType = EngineType::FourCylinder;
    }

    part def SixCylinderEngine :> Engine {
        attribute redefines engineType = EngineType::SixCylinder;
    }

    part def ManualTransmission :> Transmission {
        attribute redefines transmissionType = TransmissionType::Manual;
    }

    part def AutomaticTransmission :> Transmission {
        attribute redefines transmissionType = TransmissionType::Automatic;
    }

    part def Powertrain {
        attribute performanceModel : PerformanceModel;
        part engine : Engine;
        part transmission : Transmission;
    }

    part def StandardPowertrain :> Powertrain {
        attribute redefines performanceModel = PerformanceModel::Standard;
        part redefines engine : FourCylinderEngine;
        part redefines transmission : ManualTransmission;
    }

    part def HighPerformancePowertrain :> Powertrain {
        attribute redefines performanceModel = PerformanceModel::HighPerformance;
        part redefines engine : SixCylinderEngine;
        part redefines transmission : AutomaticTransmission;
    }

    part def Vehicle {
        part powertrain : Powertrain;
    }

    part def StandardVehicle :> Vehicle {
        part redefines powertrain : StandardPowertrain;
    }

    part def HighPerformanceVehicle :> Vehicle {
        part redefines powertrain : HighPerformancePowertrain;
    }

    part standardVehicleInstance : StandardVehicle;

    part highPerformanceVehicleInstance : HighPerformanceVehicle;
}