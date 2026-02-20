package FuelSystemModel {
  private import ScalarValues::*;

  item def Fuel {
    attribute temperature : Real;
  }

  port def FuelPort {
    item ref fuel : Fuel;
  }

  part def FuelTankAssembly {
    port supply : FuelPort;
    port returnFuel : FuelPort;
  }

  part def Engine {
    port supplyIn : FuelPort;
    port returnOut : FuelPort;
  }

  part def FuelSystem {
    part tank : FuelTankAssembly;
    part engine : Engine;

    connect tank.supply to engine.supplyIn;
    connect engine.returnOut to tank.returnFuel;
  }
}