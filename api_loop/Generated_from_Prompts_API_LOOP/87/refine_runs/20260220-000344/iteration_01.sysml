package CarSafetyCompliance {
   private import ScalarValues::*;

   part def SeatBelt {
      attribute isSafety : Boolean = true;
      attribute isMandatory : Boolean = true;
   }

   part def Airbag {
      attribute isSafety : Boolean = true;
      attribute isMandatory : Boolean = false;
   }

   part def Bumper {
      attribute isSafety : Boolean = true;
      attribute isMandatory : Boolean = true;
   }

   part def KeylessEntry {
      attribute isSafety : Boolean = false;
      attribute isMandatory : Boolean = false;
   }

   part def Wheel {
      attribute isSafety : Boolean = true;
      attribute isMandatory : Boolean = true;
   }

   part def ABS {
      attribute isSafety : Boolean = true;
      attribute isMandatory : Boolean = false;
   }

   part def Car {
      part seatBelt : SeatBelt [2];
      part airbag : Airbag [0..1];
      part bumper : Bumper;
      part keylessEntry : KeylessEntry [0..1];
      part wheel : Wheel [2];
      part abs : ABS [0..1];
   }

   package SafetyFeatures {
      part seatBelt : SeatBelt;
      part airbag : Airbag;
      part bumper : Bumper;
      part wheel : Wheel;
      part abs : ABS;
   }

   package MandatorySafetyFeatures {
      part seatBelt : SeatBelt;
      part bumper : Bumper;
      part wheel : Wheel;
   }
}