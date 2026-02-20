package CarSafetyCompliance {
    private import ScalarValues::*;

    enum def SafetyLevel {
        enum MANDATORY;
        enum OPTIONAL;
        enum NOT_REQUIRED;
    }

    attribute def SafetyMetadata {
        attribute isSafety : Boolean;
        attribute isMandatory : Boolean;
        attribute safetyLevel : SafetyLevel;
    }

    part def SeatBelt {
        attribute isSafety : Boolean = true;
        attribute isMandatory : Boolean = true;
        attribute safetyLevel : SafetyLevel = SafetyLevel::MANDATORY;
        attribute quantity : Integer = 2;
    }

    part def DriverAirbag {
        attribute isSafety : Boolean = true;
        attribute isMandatory : Boolean = false;
        attribute safetyLevel : SafetyLevel = SafetyLevel::OPTIONAL;
    }

    part def Bumper {
        attribute isSafety : Boolean = true;
        attribute isMandatory : Boolean = true;
        attribute safetyLevel : SafetyLevel = SafetyLevel::MANDATORY;
    }

    part def KeylessEntry {
        attribute isSafety : Boolean = false;
        attribute isMandatory : Boolean = false;
        attribute safetyLevel : SafetyLevel = SafetyLevel::NOT_REQUIRED;
    }

    part def Wheel {
        attribute isSafety : Boolean = false;
        attribute isMandatory : Boolean = true;
        attribute quantity : Integer = 2;
    }

    part def AntiLockBrakingSystem {
        attribute isSafety : Boolean = true;
        attribute isMandatory : Boolean = false;
        attribute safetyLevel : SafetyLevel = SafetyLevel::OPTIONAL;
    }

    part def Interior {
        part seatBelt1 : SeatBelt;
        part seatBelt2 : SeatBelt;
        part driverAirbag : DriverAirbag;
    }

    part def Body {
        part bumper : Bumper;
        part keylessEntry : KeylessEntry;
    }

    part def WheelAssembly {
        part wheel1 : Wheel;
        part wheel2 : Wheel;
        part abs : AntiLockBrakingSystem;
    }

    part def Car {
        part interior : Interior;
        part body : Body;
        part wheelAssembly : WheelAssembly;
    }

    package SafetyFeatures {
        part def SafetySeatBelt :> SeatBelt {
            attribute annotation : String = "Safety";
        }
        part def SafetyDriverAirbag :> DriverAirbag {
            attribute annotation : String = "Safety";
        }
        part def SafetyBumper :> Bumper {
            attribute annotation : String = "Safety";
        }
        part def SafetyABS :> AntiLockBrakingSystem {
            attribute annotation : String = "Safety";
        }
    }

    package MandatorySafetyFeatures {
        part def MandatorySeatBelt :> SeatBelt {
            attribute annotation : String = "Safety_Mandatory";
            attribute verifiedMandatory : Boolean = true;
        }
        part def MandatoryBumper :> Bumper {
            attribute annotation : String = "Safety_Mandatory";
            attribute verifiedMandatory : Boolean = true;
        }
    }

    part carInstance : Car {
        part interior : Interior {
            part seatBelt1 : SeatBelt;
            part seatBelt2 : SeatBelt;
            part driverAirbag : DriverAirbag;
        }
        part body : Body {
            part bumper : Bumper;
            part keylessEntry : KeylessEntry;
        }
        part wheelAssembly : WheelAssembly {
            part wheel1 : Wheel;
            part wheel2 : Wheel;
            part abs : AntiLockBrakingSystem;
        }
    }
}