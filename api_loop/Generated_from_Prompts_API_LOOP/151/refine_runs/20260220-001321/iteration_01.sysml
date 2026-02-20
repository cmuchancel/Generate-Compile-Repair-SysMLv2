package TransportationService {
  private import ScalarValues::*;

  enum def VehicleUseStatus {
    enum idle;
    enum inUse;
  }

  part def Environment {
    attribute roadCondition : String;
    attribute weather : String;
  }

  part def Driver {
    attribute licensed : Boolean;
    attribute inVehicle : Boolean;
  }

  part def Passenger {
    attribute inVehicle : Boolean;
  }

  part def GasStation {
    attribute available : Boolean;
  }

  part def Vehicle {
    attribute maxFuelCapacity : Real;
    attribute fuelLevel : Real;
    attribute fuelLowThreshold : Real = 0.1;
    attribute seats : Integer = 5;
  }

  part def TransportationSystem {
    part vehicle : Vehicle;
    part driver : Driver;
    part passengers : Passenger [0..4];
    part environment : Environment;
    part gasStation : GasStation;
    attribute status : VehicleUseStatus = idle;
  }
}