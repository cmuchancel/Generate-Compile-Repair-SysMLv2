package VehicleSystem {
   private import ScalarValues::*;

   part def Cylinder;

   part def Engine;

   part def SmallEngine : Engine {
      part cylinders : Cylinder[4];
   }

   part def LargeEngine : Engine {
      part cylinders : Cylinder[6];
   }

   part def Vehicle {
      part engine : Engine;
   }

   part def SmallVehicle : Vehicle {
      part engine : SmallEngine;
   }

   part def LargeVehicle : Vehicle {
      part engine : LargeEngine;
   }

   part smallVehicleInstance : SmallVehicle;
   part largeVehicleInstance : LargeVehicle;
}