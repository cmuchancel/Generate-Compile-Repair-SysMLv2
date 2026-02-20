package CarSafetyCompliance {
    private import ScalarValues::*;

    enum def SafetyStatus {
        enum Mandatory;
        enum Optional;
    }

    part def Car {
        part body : Body;
        part interior : Interior;
        part wheelAssembly : WheelAssembly;
    }

    part def Body {
        attribute bumperSafety : SafetyStatus = SafetyStatus::Mandatory;
        attribute keylessEntrySafety : SafetyStatus = SafetyStatus::Optional;
    }

    part def Interior {
        attribute seatBeltCount : Integer = 2;
        attribute seatBeltSafety : SafetyStatus = SafetyStatus::Mandatory;
        attribute driverAirbagSafety : SafetyStatus = SafetyStatus::Optional;
    }

    part def WheelAssembly {
        attribute wheelCount : Integer = 2;
        attribute absSafety : SafetyStatus = SafetyStatus::Optional;
    }

    package SafetyFeatures {
        part bodyBumper : Body {
            attribute bumperSafety : SafetyStatus = SafetyStatus::Mandatory;
        }

        part interiorSeatBelts : Interior {
            attribute seatBeltSafety : SafetyStatus = SafetyStatus::Mandatory;
        }

        part wheelAssemblyABS : WheelAssembly {
            attribute absSafety : SafetyStatus = SafetyStatus::Optional;
        }
    }

    package MandatorySafetyFeatures {
        part bodyBumper : Body {
            attribute bumperSafety : SafetyStatus = SafetyStatus::Mandatory;
        }

        part interiorSeatBelts : Interior {
            attribute seatBeltSafety : SafetyStatus = SafetyStatus::Mandatory;
        }
    }
}