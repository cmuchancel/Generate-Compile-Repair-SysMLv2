package PowertrainSystem {
    private import ScalarValues::*;

    enum def PowerSourceType {
        enum InternalCombustionEngine;
        enum ElectricMotor;
        enum Hybrid;
    }

    part def TorqueGenerator {
        attribute torqueOutput : Real;
        attribute efficiency : Real;
    }

    part def PowerUnit {
        attribute powerOutput : Real;
        attribute torqueOutput : Real;
        attribute sourceType : PowerSourceType;
    }

    part def LogicalLayer {
        part torqueGenerator : TorqueGenerator;
    }

    part def PhysicalLayer {
        part powerUnit : PowerUnit;
    }

    part def PowertrainSystem {
        part logicalLayer : LogicalLayer;
        part physicalLayer : PhysicalLayer;

        connect logicalLayer.torqueGenerator to physicalLayer.powerUnit;
    }
}