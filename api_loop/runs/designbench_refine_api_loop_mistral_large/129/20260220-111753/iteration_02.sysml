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

    part FrontSeatBelt :> SafetyComponent {
        attribute componentName : String = "Front Seat Belt";
        attribute componentCategory : SafetyCategory = SafetyCategory::Mandatory;
    }

    part Bumper :> SafetyComponent {
        attribute componentName : String = "Bumper";
        attribute componentCategory : SafetyCategory = SafetyCategory::Mandatory;
    }

    part DriversAirbag :> SafetyComponent {
        attribute componentName : String = "Driver's Airbag";
        attribute componentCategory : SafetyCategory = SafetyCategory::Optional;
    }

    part AntiLockBrakingSystem :> SafetyComponent {
        attribute componentName : String = "Anti-lock Braking System";
        attribute componentCategory : SafetyCategory = SafetyCategory::Optional;
    }

    part AlarmSystem :> AntiTheftComponent {
        attribute componentName : String = "Alarm System";
    }

    part KeylessEntrySystem :> AntiTheftComponent {
        attribute componentName : String = "Keyless Entry System";
    }

    part Immobilizer :> CombinedComponent {
        attribute componentName : String = "Immobilizer";
    }

}