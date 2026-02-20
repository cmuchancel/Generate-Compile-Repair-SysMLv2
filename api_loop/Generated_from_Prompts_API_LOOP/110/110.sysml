package FuelDeliverySystem {
   private import ScalarValues::*;

   item def Fuel;

   port def FuelPort;

   part def StorageTank {
      attribute maxCapacity : Real;
      attribute currentLevel : Real;
      port outFuel : FuelPort;
   }

   part def FuelPump {
      attribute flowRate : Real;
      port inFuel : FuelPort;
      port outFuel : FuelPort;
   }

   part def VehicleFuelTank {
      attribute maxCapacity : Real;
      attribute currentLevel : Real;
      port inFuel : FuelPort;
   }

   part def FuelDeliverySystem {
      part storage : StorageTank;
      part pump : FuelPump;
      part vehicleTank : VehicleFuelTank;
      connect storage.outFuel to pump.inFuel;
      connect pump.outFuel to vehicleTank.inFuel;
   }
}