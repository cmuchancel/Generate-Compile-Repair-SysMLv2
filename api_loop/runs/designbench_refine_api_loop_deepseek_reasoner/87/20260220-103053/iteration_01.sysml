package CarSafetyCompliance {
    private import ScalarValues::*;

    enum def SafetyClassification {
        Safety,
        NonSafety
    }

    part def Component {
        attribute safetyClass : SafetyClassification;
        attribute isMandatory : Boolean;
    }

    part def SeatBelt {
        attribute safetyClass : SafetyClassification = SafetyClassification::Safety;
        attribute isMandatory : Boolean = true;
    }

    part def Airbag {
        attribute safetyClass : SafetyClassification = SafetyClassification::Safety;
        attribute isMandatory : Boolean = false;
    }

    part def Bumper {
        attribute safetyClass : SafetyClassification = SafetyClassification::Safety;
        attribute isMandatory : Boolean = true;
    }

    part def KeylessEntry {
        attribute safetyClass : SafetyClassification = SafetyClassification::NonSafety;
        attribute isMandatory : Boolean = false;
    }

    part def Wheel {
        attribute safetyClass : SafetyClassification = SafetyClassification::NonSafety;
        attribute isMandatory : Boolean = true;
    }

    part def ABS {
        attribute safetyClass : SafetyClassification = SafetyClassification::Safety;
        attribute isMandatory : Boolean = false;
    }

    part def Interior {
        part seatBelt1 : SeatBelt;
        part seatBelt2 : SeatBelt;
        part driverAirbag : Airbag;
    }

    part def Body {
        part bumper : Bumper;
        part keylessEntry : KeylessEntry;
    }

    part def WheelAssembly {
        part wheel1 : Wheel;
        part wheel2 : Wheel;
        part abs : ABS[0..1];
    }

    part def Car {
        part interior : Interior;
        part body : Body;
        part wheelAssembly : WheelAssembly;
    }

    package SafetyFeatures {
        part interiorSeatBelt1 : SeatBelt;
        part interiorSeatBelt2 : SeatBelt;
        part driverAirbag : Airbag;
        part bumper : Bumper;
        part abs : ABS;
    }

    package MandatorySafetyFeatures {
        part interiorSeatBelt1 : SeatBelt;
        part interiorSeatBelt2 : SeatBelt;
        part bumper : Bumper;
    }
}