package FuelManagement {
   private import ScalarValues::*;

   item def FuelFlow {
      attribute rate : Real;
   }

   item def TemperatureInfo {
      attribute temperature : Real;
   }

   item def FlowStatus {
      attribute supplyRate : Real;
      attribute returnRate : Real;
      attribute temperature : Real;
   }

   port def FuelSupplyPort {
      item flow : FuelFlow;
   }

   port def FuelReturnPort {
      item flow : FuelFlow;
   }

   port def TemperaturePort {
      item temp : TemperatureInfo;
   }

   port def StatusPort {
      item status : FlowStatus;
   }

   part def FuelTankAssembly {
      port supplyOut : FuelSupplyPort;
      port returnIn : FuelReturnPort;
      port temperature : TemperaturePort;
      port statusOut : StatusPort;
      attribute currentSupplyRate : Real;
      attribute currentReturnRate : Real;
      attribute currentTemperature : Real;
   }

   part def Engine {
      port fuelIn : FuelSupplyPort;
      port returnOut : FuelReturnPort;
      port temperature : TemperaturePort;
   }

   part def FuelManagementSystem {
      part tank : FuelTankAssembly;
      part engine : Engine;
      port statusOut : StatusPort;

      connect tank.supplyOut to engine.fuelIn;
      connect engine.returnOut to tank.returnIn;
      connect tank.temperature to engine.temperature;
      connect tank.statusOut to statusOut;
   }
}