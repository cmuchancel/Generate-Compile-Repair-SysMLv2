package FuelSystemModel {
  private import ScalarValues::*;

  item def Fuel;
  item def Temperature;

  port def FuelPort;
  port def TempPort;

  part def FuelTank {
    port outlet: FuelPort;
    port inlet: FuelPort;
    port tempOut: TempPort;
    attribute fuelTemperature: Temperature;
  }

  part def FuelPump {
    port inlet: FuelPort;
    port outlet: FuelPort;
    port returnIn: FuelPort;
  }

  part def FuelTankAssembly {
    part tank: FuelTank;
    part pump: FuelPump;
    port outlet: FuelPort;
    port returnPort: FuelPort;
    port tempOut: TempPort;

    connect tank.outlet to pump.inlet;
    connect pump.outlet to outlet;
    connect returnPort to pump.returnIn;
    connect pump.returnIn to tank.inlet;
    connect tank.tempOut to tempOut;
  }

  part def Engine {
    port fuelIn: FuelPort;
    port returnOut: FuelPort;
  }

  part def FuelSystem {
    part assembly: FuelTankAssembly;
    part engine: Engine;
    port tempOut: TempPort;

    connect assembly.outlet to engine.fuelIn;
    connect engine.returnOut to assembly.returnPort;
    connect assembly.tempOut to tempOut;
  }
}