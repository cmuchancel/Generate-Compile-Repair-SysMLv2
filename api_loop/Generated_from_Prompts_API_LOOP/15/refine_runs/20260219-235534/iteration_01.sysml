package FuelSystemModel {
  private import ScalarValues::*;

  item def Fuel;

  item def TemperatureInfo {
    attribute value : Real;
  }

  port def FuelPort {
    item fuel : Fuel;
  }

  port def TemperaturePort {
    item temp : TemperatureInfo;
  }

  part def FuelTankAssembly {
    port supply : FuelPort;
    port return : FuelPort;
    port tempOut : TemperaturePort;
  }

  part def Engine {
    port fuelIn : FuelPort;
    port fuelReturn : FuelPort;
    port tempOut : TemperaturePort;
  }

  part def TemperatureMonitor {
    port tempIn1 : TemperaturePort;
    port tempIn2 : TemperaturePort;
    port tempOut : TemperaturePort;
  }

  part def VehicleFuelSystem {
    part tank : FuelTankAssembly;
    part engine : Engine;
    part monitor : TemperatureMonitor;
    port systemTempOut : TemperaturePort;

    connect tank.supply to engine.fuelIn;
    connect engine.fuelReturn to tank.return;
    connect tank.tempOut to monitor.tempIn1;
    connect engine.tempOut to monitor.tempIn2;
    connect monitor.tempOut to systemTempOut;
  }
}