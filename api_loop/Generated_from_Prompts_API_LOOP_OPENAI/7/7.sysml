package VehicleSystem {
   private import ScalarValues::*;

   part def Cylinder;

   part def Engine;

   part def SmallEngine {
      part cylinders : Cylinder[4];
   }

   part def LargeEngine {
      part cylinders : Cylinder[6];
   }

   part def Vehicle {
      part engine : Engine;
   }

   part def SmallVehicle {
      part engine : SmallEngine;
   }

   part def LargeVehicle {
      part engine : LargeEngine;
   }

   part smallVehicleInstance : SmallVehicle;
   part largeVehicleInstance : LargeVehicle;
}