package VehicleSafetyAntiTheftCatalogue {
    private import ScalarValues::*;

    enum def ComponentCategory {
        enum Safety;
        enum AntiTheft;
        enum Both;
    }

    enum def RequirementLevel {
        enum Mandatory;
        enum Optional;
    }

    part def Vehicle {
        part body : VehicleBody;
        part interior : VehicleInterior;
        part security : VehicleSecurity;
        part braking : VehicleBraking;
    }

    part def VehicleBody {
        part bumper : Bumper;
    }

    part def VehicleInterior {
        part frontSeatBelts : FrontSeatBelts;
        part driversAirbag : DriversAirbag;
    }

    part def VehicleSecurity {
        part alarmSystem : AlarmSystem;
        part keylessEntrySystem : KeylessEntrySystem;
    }

    part def VehicleBraking {
        part antiLockBrakingSystem : AntiLockBrakingSystem;
    }

    part def Bumper {
        attribute category : ComponentCategory = ComponentCategory::Safety;
        attribute requirement : RequirementLevel = RequirementLevel::Mandatory;
    }

    part def FrontSeatBelts {
        attribute category : ComponentCategory = ComponentCategory::Safety;
        attribute requirement : RequirementLevel = RequirementLevel::Mandatory;
    }

    part def DriversAirbag {
        attribute category : ComponentCategory = ComponentCategory::Safety;
        attribute requirement : RequirementLevel = RequirementLevel::Optional;
    }

    part def AntiLockBrakingSystem {
        attribute category : ComponentCategory = ComponentCategory::Safety;
        attribute requirement : RequirementLevel = RequirementLevel::Optional;
    }

    part def AlarmSystem {
        attribute category : ComponentCategory = ComponentCategory::AntiTheft;
        attribute requirement : RequirementLevel = RequirementLevel::Optional;
    }

    part def KeylessEntrySystem {
        attribute category : ComponentCategory = ComponentCategory::AntiTheft;
        attribute requirement : RequirementLevel = RequirementLevel::Optional;
    }
}