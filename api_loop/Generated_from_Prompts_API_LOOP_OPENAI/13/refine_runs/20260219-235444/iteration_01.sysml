package FuelSupplySystem {
  private import ScalarValues::*;

  item def Fuel;
  item def Temperature {
    attribute value : Real;
  }

  port def FuelPort {
    item fuel : Fuel;
  }

  port def TemperaturePort {
    item temperature : Temperature;
  }

  part def FuelTank {
    port supplyOut : FuelPort;
    port returnIn : FuelPort;
  }

  part def FuelPump {
    port inlet : FuelPort;
    port outlet : FuelPort;
  }

  part def FuelTemperatureSensor {
    port tempOut : TemperaturePort;
  }

  part def FuelTankAssembly {
    part tank : FuelTank;
    part pump : FuelPump;
    part tempSensor : FuelTemperatureSensor;
    connect tank.supplyOut to pump.inlet;
  }

  part def ReturnLine {
    port fromUnits : FuelPort;
    port toTank : FuelPort;
  }

  part def FuelSystem {
    part assembly : FuelTankAssembly;
    part returnLine : ReturnLine;
    port supply : FuelPort;
    port returnIn : FuelPort;
    port tempOut : TemperaturePort;
    connect assembly.pump.outlet to supply;
    connect returnIn to returnLine.fromUnits;
    connect returnLine.toTank to assembly.tank.returnIn;
    connect assembly.tempSensor.tempOut to tempOut;
  }

  part def FuelConsumingUnit {
    port inFuel : FuelPort;
    port returnFuel : FuelPort;
  }

  part def TemperatureMonitor {
    port tempIn : TemperaturePort;
  }

  part def Vehicle {
    part fuelSystem : FuelSystem;
    part consumers : FuelConsumingUnit;
    part tempMonitor : TemperatureMonitor;
    connect fuelSystem.supply to consumers.inFuel;
    connect consumers.returnFuel to fuelSystem.returnIn;
    connect fuelSystem.tempOut to tempMonitor.tempIn;
  }
}