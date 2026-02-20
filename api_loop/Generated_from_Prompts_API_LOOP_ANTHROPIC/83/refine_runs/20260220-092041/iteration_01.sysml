package PowertrainSystem {
    private import ScalarValues::*;

    enum def PowertrainLayer {
        enum logical;
        enum physical;
    }

    item def Power;
    item def Torque;

    port def PowerPort {
        attribute powerLevel : Real;
    }

    port def TorquePort {
        attribute torqueLevel : Real;
    }

    part def LogicalPowerProvider {
        attribute layer : PowertrainLayer = PowertrainLayer::logical;
        port powerOut : PowerPort;
    }

    part def TorqueGenerator {
        attribute layer : PowertrainLayer = PowertrainLayer::logical;
        port powerIn : PowerPort;
        port torqueOut : TorquePort;
    }

    part def PhysicalPowerUnit {
        attribute layer : PowertrainLayer = PowertrainLayer::physical;
        port powerOut : PowerPort;
        port torqueOut : TorquePort;
    }

    part def Engine {
        attribute layer : PowertrainLayer = PowertrainLayer::physical;
        port powerOut : PowerPort;
        port torqueOut : TorquePort;
    }

    part def LogicalLayer {
        part powerProvider : LogicalPowerProvider;
        part torqueGen : TorqueGenerator;
        connect powerProvider.powerOut to torqueGen.powerIn;
    }

    part def PhysicalLayer {
        part powerUnit : PhysicalPowerUnit;
        part engine : Engine;
        connect powerUnit.powerOut to engine.powerOut;
        connect powerUnit.torqueOut to engine.torqueOut;
    }

    part def PowertrainSystem {
        attribute systemName : String = "PowertrainSystem";
        part logical : LogicalLayer;
        part physical : PhysicalLayer;
        connect logical.torqueGen.torqueOut to physical.engine.torqueOut;
    }

    part powertrain : PowertrainSystem;
}