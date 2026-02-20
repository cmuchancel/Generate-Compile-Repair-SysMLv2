package FuelSystemModel {
  private import ScalarValues::*;

  item def Fuel {
    attribute temperature : Real;
  }

  port def FuelPort {
    item fuel : Fuel;
  }

  part def FuelTankAssembly {
    port supply : FuelPort;
    port return : FuelPort;
  }

  part def Engine {
    port supplyIn : FuelPort;
    port returnOut : FuelPort;
  }

  part def FuelSystem {
    part tank : FuelTankAssembly;
    part engine : Engine;

    connect tank.supply to engine.supplyIn;
    connect engine.returnOut to tank.return;
  }
}