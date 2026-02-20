package VehicleCompliance {
    private import ScalarValues::*;

    enum def SafetyClassification {
        enum Mandatory;
        enum Optional;
    }

    enum def SecurityClassification {
        enum Mandatory;
        enum Optional;
    }

    part def SeatBelt;
    part def Airbag;
    part def AntiTheftAlarm;
    part def Bumper;
    part def KeylessEntrySystem;
    part def Wheel;
    part def AntiLockBrakingSystem;

    part def VehicleInterior {
        attribute seatBelt1 : SeatBelt;
        attribute seatBelt2 : SeatBelt;
        attribute airbag : Airbag;
        attribute antiTheftAlarm : AntiTheftAlarm;
        attribute safetyClassification : SafetyClassification = SafetyClassification::Mandatory;
        attribute securityClassification : SecurityClassification = SecurityClassification::Mandatory;
    }

    part def VehicleBody {
        attribute bumper : Bumper;
        attribute keylessEntrySystem : KeylessEntrySystem;
        attribute safetyClassification : SafetyClassification = SafetyClassification::Mandatory;
        attribute securityClassification : SecurityClassification = SecurityClassification::Mandatory;
    }

    part def VehicleChassis {
        attribute wheel1 : Wheel;
        attribute wheel2 : Wheel;
        attribute abs1 : AntiLockBrakingSystem;
        attribute abs2 : AntiLockBrakingSystem;
        attribute safetyClassification : SafetyClassification = SafetyClassification::Optional;
    }

    part def Vehicle {
        part interior : VehicleInterior;
        part body : VehicleBody;
        part chassis : VehicleChassis;
    }
}