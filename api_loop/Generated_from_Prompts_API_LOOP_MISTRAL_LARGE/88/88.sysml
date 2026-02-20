package VehicleSafetyCompliance {
    private import ScalarValues::*;

    enum def SafetyCategory {
        enum Mandatory;
        enum Optional;
    }

    part def Vehicle {
        part interior : VehicleInterior;
        part body : VehicleBody;
        part chassis : VehicleChassis;
    }

    part def VehicleInterior {
        attribute seatBelts : SeatBelt[2];
        attribute driverAirbag : Airbag;
    }

    part def VehicleBody {
        attribute bumper : Bumper;
        attribute keylessEntry : KeylessEntrySystem;
    }

    part def VehicleChassis {
        attribute tires : Tire[4];
        attribute abs : AntiLockBrakingSystem;
    }

    part def SeatBelt {
        attribute safetyCategory : SafetyCategory = SafetyCategory::Mandatory;
    }

    part def Airbag {
        attribute safetyCategory : SafetyCategory = SafetyCategory::Optional;
    }

    part def Bumper {
        attribute safetyCategory : SafetyCategory = SafetyCategory::Mandatory;
    }

    part def KeylessEntrySystem {
        attribute safetyCategory : SafetyCategory = SafetyCategory::Optional;
    }

    part def Tire {
    }

    part def AntiLockBrakingSystem {
        attribute safetyCategory : SafetyCategory = SafetyCategory::Optional;
    }
}