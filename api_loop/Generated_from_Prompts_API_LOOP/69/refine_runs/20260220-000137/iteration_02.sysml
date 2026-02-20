package VehicleSystem {
  private import ScalarValues::*;

  port def TorquePort {
    attribute torque : Real;
  }

  part def Engine {
    port torqueOut : TorquePort;
    attribute maxTorque : Real = 300;
    attribute requiredTorque1000 : Real = 180;
    attribute deliveredTorque1000 : Real = 180;
    attribute requiredTorque2000 : Real = 220;
    attribute deliveredTorque2000 : Real = 220;
    attribute requiredTorque3000 : Real = 200;
    attribute deliveredTorque3000 : Real = 200;
  }

  part def Transmission {
    port torqueIn : TorquePort;
  }

  part def Clutch {
    port torqueIn : TorquePort;
    port torqueOut : TorquePort;
  }

  part def FuelTank {
    attribute fuelMassFull : Real = 100;
    attribute fuelMassEmpty : Real = 0;
  }

  part def Vehicle {
    attribute dryMass : Real = 1400;
    attribute fuelMassFull : Real = 100;
    attribute fuelMassEmpty : Real = 0;
    attribute totalMassFull : Real = 1500;
    attribute totalMassEmpty : Real = 1400;
    attribute maxTotalMassFull : Real = 2000;
    attribute maxTotalMassEmpty : Real = 1500;
    part engine : Engine;
    part clutch : Clutch;
    part transmission : Transmission;
    part fuelTank : FuelTank;
    connect engine.torqueOut to clutch.torqueIn;
    connect clutch.torqueOut to transmission.torqueIn;
  }
}