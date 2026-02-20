package PowerSystem {
  private import ScalarValues::*;

  item def Torque;
  item def Power;

  port def TorquePort {
    ref item torque : Torque;
  }

  port def PowerPort {
    ref item power : Power;
  }

  part def TorqueGenerator {
    port torqueOut : TorquePort;
  }

  part def Engine {
    port torqueOut : TorquePort;
  }

  part def Powertrain {
    part engine : Engine;
    port torqueOut : TorquePort;
    port powerOut : PowerPort;
    connect engine.torqueOut to torqueOut;
  }

  part def PowerTransmissionSystem {
    part torqueGenerator : TorqueGenerator;
    part powertrain : Powertrain;
    port torqueOut : TorquePort;
    port powerOut : PowerPort;
    connect torqueGenerator.torqueOut to powertrain.engine.torqueOut;
    connect powertrain.torqueOut to torqueOut;
    connect powertrain.powerOut to powerOut;
  }
}