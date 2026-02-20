package TransportSystem {
  private import ScalarValues::*;
  
  enum def EnvironmentCondition {
    enum Clear;
    enum Rain;
    enum Snow;
    enum Wind;
  }
  
  port def ControlPort;
  port def FuelPort;
  port def OccupancyPort;
  
  part def Driver {
    attribute name : String;
    attribute licensed : Boolean = true;
    port controlPort : ControlPort;
    port occupancyPort : OccupancyPort;
  }
  
  part def Passenger {
    attribute name : String;
    port occupancyPort : OccupancyPort;
  }
  
  part def Vehicle {
    attribute capacity : Integer = 5;
    attribute fuelLevel : Real;
    port controlPort : ControlPort;
    port fuelPort : FuelPort;
    port occupancyPort : OccupancyPort;
  }
  
  part def GasStation {
    attribute fuelAvailable : Boolean = true;
    port fuelPort : FuelPort;
  }
  
  part def Environment {
    attribute condition : EnvironmentCondition;
  }
  
  part def Trip {
    attribute startPoint : String;
    attribute destination : String;
  }
  
  part def TransportationSystem {
    attribute maxPassengers : Integer = 4;
    part driver : Driver;
    part passengers : Passenger [0..4];
    part vehicle : Vehicle;
    part gasStation : GasStation;
    part environment : Environment;
    part trip : Trip;
    connect driver.controlPort to vehicle.controlPort;
    connect driver.occupancyPort to vehicle.occupancyPort;
    connect vehicle.fuelPort to gasStation.fuelPort;
  }
}