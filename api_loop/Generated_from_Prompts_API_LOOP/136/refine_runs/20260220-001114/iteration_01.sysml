package AutomotiveTires {
   private import ScalarValues::*;
   
   part def Tire {
      attribute manufacturer : String;
      attribute rimDiameter : Real;
      attribute width : Integer;
   }
   
   part MichelinFrenchTire : Tire {
      attribute manufacturer : String = "Michelin";
      attribute rimDiameter : Real = 18.0;
      attribute width : Integer = 245;
   }
   
   part BridgestoneTire : Tire {
      attribute manufacturer : String = "Bridgestone";
      attribute rimDiameter : Real = 17.0;
      attribute width : Integer = 225;
   }
}