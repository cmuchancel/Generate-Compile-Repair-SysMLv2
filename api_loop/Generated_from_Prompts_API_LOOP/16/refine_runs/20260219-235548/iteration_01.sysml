package VehicleFuelManagement {
  private import ScalarValues::*;

  port def FuelPort {
    attribute fuelTemperature : Real;
  }

  part def FuelTankAssembly {
    port outlet : FuelPort;
    port returnIn : FuelPort;
  }

  part def EngineUnit {
    port inlet : FuelPort;
    port returnOut : FuelPort;
  }

  part def FuelSystem {
    part tank : FuelTankAssembly;
    part engine : EngineUnit;

    connect tank.outlet to engine.inlet;
    connect engine.returnOut to tank.returnIn;
  }
}