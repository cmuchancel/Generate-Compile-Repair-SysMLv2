package FuelSystemModel {
  private import ScalarValues::*;
  
  item def Fuel;
  
  port def FuelSupplyPort;
  port def FuelReturnPort;
  
  part def FuelTankAssembly {
    port supplyOut : FuelSupplyPort;
    port returnIn : FuelReturnPort;
    attribute fuelTemperature : Real;
  }
  
  part def Engine {
    port supplyIn : FuelSupplyPort;
    port returnOut : FuelReturnPort;
    attribute fuelTemperature : Real;
  }
  
  part def FuelSystem {
    part tank : FuelTankAssembly;
    part engine : Engine;
    connect tank.supplyOut to engine.supplyIn;
    connect engine.returnOut to tank.returnIn;
    attribute monitoredTemperature : Real;
  }
  
  part fuelSystem : FuelSystem;
}