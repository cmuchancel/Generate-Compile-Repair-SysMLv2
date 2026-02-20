package VehicleSystem {
  private import ScalarValues::*;

  port def TorquePort {
    attribute torque : Real;
    attribute speed : Real;
  }

  part def Engine {
    port torqueOut : TorquePort;
    attribute maxTorque : Real = 300;
    attribute ratedSpeed : Real = 4000;
  }

  part def Clutch {
    port engineSide : TorquePort;
    port transSide : TorquePort;
  }

  part def Transmission {
    port input : TorquePort;
  }

  part def Vehicle {
    attribute curbMass : Real = 1450;
    attribute fuelMass : Real = 50;
    attribute totalMass : Real = 1500;
    attribute maxMassLimit : Real = 2000;

    part engine : Engine;
    part clutch : Clutch;
    part transmission : Transmission;

    connect engine.torqueOut to clutch.engineSide;
    connect clutch.transSide to transmission.input;
  }
}