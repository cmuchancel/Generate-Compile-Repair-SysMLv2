package VehicleSystem {
  private import ScalarValues::*;

  port def TorquePort {
    attribute torque : Torque;
    attribute speed : Speed;
  }

  part def Engine {
    port torqueOut : TorquePort;
    attribute maxTorque : Torque = 300;
    attribute ratedSpeed : Speed = 4000;
  }

  part def Clutch {
    port engineSide : TorquePort;
    port transSide : TorquePort;
  }

  part def Transmission {
    port input : TorquePort;
  }

  part def Vehicle {
    attribute curbMass : Mass = 1450;
    attribute fuelMass : Mass = 50;
    attribute totalMass : Mass = 1500;
    attribute maxMassLimit : Mass = 2000;

    part engine : Engine;
    part clutch : Clutch;
    part transmission : Transmission;

    connect engine.torqueOut to clutch.engineSide;
    connect clutch.transSide to transmission.input;
  }
}