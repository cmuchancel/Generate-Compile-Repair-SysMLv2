package VehicleSafetyAndAntiTheft {

    private import ScalarValues::*;

    enum def SafetyCategory {
        enum Mandatory;
        enum Optional;
        enum BothSafetyAndAntiTheft;
    }

    enum def ComponentType {
        enum Safety;
        enum AntiTheft;
        enum Both;
    }

    part def Vehicle {
        attribute safetyComponents : SafetyComponent[*];
        attribute antiTheftComponents : AntiTheftComponent[*];
        attribute combinedComponents : CombinedComponent[*];
    }

    part def SafetyComponent {
        attribute name : String;
        attribute category : SafetyCategory = SafetyCategory::Optional;
    }

    part def AntiTheftComponent {
        attribute name : String;
    }

    part def CombinedComponent {
        attribute name : String;
    }

    part def FrontSeatBelt : SafetyComponent {
        attribute name : String = "Front Seat Belt";
        attribute category : SafetyCategory = SafetyCategory::Mandatory;
    }

    part def Bumper : SafetyComponent {
        attribute name : String = "Bumper";
        attribute category : SafetyCategory = SafetyCategory::Mandatory;
    }

    part def DriversAirbag : SafetyComponent {
        attribute name : String = "Driver's Airbag";
        attribute category : SafetyCategory = SafetyCategory::Optional;
    }

    part def AntiLockBrakingSystem : SafetyComponent {
        attribute name : String = "Anti-lock Braking System";
        attribute category : SafetyCategory = SafetyCategory::Optional;
    }

    part def AlarmSystem : AntiTheftComponent {
        attribute name : String = "Alarm System";
    }

    part def KeylessEntrySystem : AntiTheftComponent {
        attribute name : String = "Keyless Entry System";
    }

    part def Immobilizer : CombinedComponent {
        attribute name : String = "Immobilizer";
    }

}