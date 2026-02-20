package FuelSystemModel
  private import ScalarValues::*;
  
  item def Fuel;
  
  port def FuelInterface;
  
  part def FuelTank {
    port supply : FuelInterface;
    port return : FuelInterface;
  }
  
  part def Engine {
    port supply : FuelInterface;
    port return : FuelInterface;
  }
  
  part def FuelSystem {
    part tank : FuelTank;
    part engine : Engine;
    attribute fuelTemperature : Temperature;
    attribute fuelFlowEnabled : Boolean;
    connect tank.supply to engine.supply;
    connect engine.return to tank.return;
  }
end package