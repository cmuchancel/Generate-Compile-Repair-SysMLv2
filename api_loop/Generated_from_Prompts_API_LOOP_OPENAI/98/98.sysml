package FuelSystemModel {
  private import ScalarValues::*;

  item def Fuel;

  port def FuelInterface;

  part def FuelTank {
    port supply : FuelInterface;
    port returnPort : FuelInterface;
  }

  part def Engine {
    port supply : FuelInterface;
    port returnPort : FuelInterface;
  }

  part def FuelSystem {
    part tank : FuelTank;
    part engine : Engine;
    attribute fuelTemperature : Real;
    attribute fuelFlowEnabled : Boolean;
    connect tank.supply to engine.supply;
    connect engine.returnPort to tank.returnPort;
  }
}