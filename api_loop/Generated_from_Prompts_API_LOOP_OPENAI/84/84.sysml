package PowerSystemModel {
  private import ScalarValues::*;

  part def PowerTransmissionSystem {
    attribute providesPower : Boolean = true;
    attribute generatesTorque : Boolean = true;
  }

  part def GeneratingTorqueAction {
    attribute generatesTorque : Boolean = true;
    attribute enablesPowerTransfer : Boolean = true;
  }

  part def TorqueGenerator {
    part generatingTorqueAction : GeneratingTorqueAction;
    attribute deliversPower : Boolean = true;
    attribute generatesTorque : Boolean = true;
  }

  part def LogicalModel {
    part torqueGenerator : TorqueGenerator;
    attribute deliversPower : Boolean = true;
    attribute generatesTorque : Boolean = true;
  }

  part def Engine {
    attribute generatesTorque : Boolean = true;
  }

  part def Powertrain {
    part engine : Engine;
    attribute providesPower : Boolean = true;
  }

  part def PhysicalModel {
    part powertrain : Powertrain;
  }

  part def VehicleSystem {
    part powerTransmissionSystem : PowerTransmissionSystem;
    part logicalModel : LogicalModel;
    part physicalModel : PhysicalModel;
    attribute managesPowerAndTorque : Boolean = true;
    connect logicalModel.torqueGenerator to physicalModel.powertrain;
    connect logicalModel.torqueGenerator.generatingTorqueAction to physicalModel.powertrain.engine;
  }
}