package PowertrainModel {
  private import ScalarValues::*;

  item def Power;
  item def Torque;

  port def PowerPort {
    item power : Power;
  }

  port def TorquePort {
    item torque : Torque;
  }

  part def PowerProvider {
    port powerOut : PowerPort;
  }

  part def TorqueGenerator {
    port powerIn : PowerPort;
    port torqueOut : TorquePort;
  }

  part def PowerUnit {
    port powerOut : PowerPort;
    port torqueOut : TorquePort;
  }

  part def LogicalLayer {
    part powerProvider : PowerProvider;
    part torqueGenerator : TorqueGenerator;
    connect powerProvider.powerOut to torqueGenerator.powerIn;
  }

  part def PhysicalLayer {
    part powerUnit : PowerUnit;
  }

  part def PowertrainSystem {
    part logicalLayer : LogicalLayer;
    part physicalLayer : PhysicalLayer;

    connect logicalLayer.torqueGenerator.torqueOut to physicalLayer.powerUnit.torqueOut;
    connect logicalLayer.torqueGenerator.powerIn to physicalLayer.powerUnit.powerOut;
  }
}