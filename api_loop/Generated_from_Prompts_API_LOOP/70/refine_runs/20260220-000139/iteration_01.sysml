package VehicleSystem {
   private import ScalarValues::*;

   item def TorqueItem {
      attribute value : Real;
   }

   port def ClutchPort {
      item torque : TorqueItem;
   }

   part def Engine {
      port clutchOut : ClutchPort;
      attribute torqueAt1000rpm : Real = 200.0;
      attribute torqueAt2000rpm : Real = 250.0;
      attribute torqueAt3000rpm : Real = 300.0;
   }

   part def Transmission {
      port clutchIn : ClutchPort;
   }

   part def Vehicle {
      attribute emptyMass : Real = 1400.0;
      attribute fueledMass : Real = 1800.0;
      part engine : Engine;
      part transmission : Transmission;
      connect engine.clutchOut to transmission.clutchIn;
   }

   part vehicleInstance : Vehicle;
}