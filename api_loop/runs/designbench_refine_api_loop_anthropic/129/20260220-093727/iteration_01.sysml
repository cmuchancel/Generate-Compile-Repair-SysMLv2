package VehicleSafetyAntiTheftCatalogue {
    private import ScalarValues::*;

    enum def ComponentCategory {
        enum SAFETY;
        enum ANTI_THEFT;
        enum SAFETY_AND_ANTI_THEFT;
    }

    enum def MandatoryStatus {
        enum MANDATORY;
        enum OPTIONAL;
    }

    part def SafetyComponent {
        attribute componentName : String;
        attribute category : ComponentCategory;
        attribute mandatoryStatus : MandatoryStatus;
        attribute affectsSafety : Boolean;
        attribute affectsAntiTheft : Boolean;
    }

    part def AntiTheftComponent {
        attribute componentName : String;
        attribute category : ComponentCategory;
        attribute mandatoryStatus : MandatoryStatus;
        attribute affectsSafety : Boolean;
        attribute affectsAntiTheft : Boolean;
    }

    part def DualPurposeComponent {
        attribute componentName : String;
        attribute category : ComponentCategory;
        attribute mandatoryStatus : MandatoryStatus;
        attribute affectsSafety : Boolean;
        attribute affectsAntiTheft : Boolean;
    }

    part def BodyInterior {
        attribute sectionName : String;

        part frontSeatBelt : SafetyComponent {
            attribute componentName : String = "Front Seat Belt";
            attribute category : ComponentCategory = ComponentCategory::SAFETY;
            attribute mandatoryStatus : MandatoryStatus = MandatoryStatus::MANDATORY;
            attribute affectsSafety : Boolean = true;
            attribute affectsAntiTheft : Boolean = false;
        }

        part bumper : SafetyComponent {
            attribute componentName : String = "Bumper";
            attribute category : ComponentCategory = ComponentCategory::SAFETY;
            attribute mandatoryStatus : MandatoryStatus = MandatoryStatus::MANDATORY;
            attribute affectsSafety : Boolean = true;
            attribute affectsAntiTheft : Boolean = false;
        }

        part driverAirbag : SafetyComponent {
            attribute componentName : String = "Driver Airbag";
            attribute category : ComponentCategory = ComponentCategory::SAFETY;
            attribute mandatoryStatus : MandatoryStatus = MandatoryStatus::OPTIONAL;
            attribute affectsSafety : Boolean = true;
            attribute affectsAntiTheft : Boolean = false;
        }

        part antiLockBrakingSystem : SafetyComponent {
            attribute componentName : String = "Anti-Lock Braking System";
            attribute category : ComponentCategory = ComponentCategory::SAFETY;
            attribute mandatoryStatus : MandatoryStatus = MandatoryStatus::OPTIONAL;
            attribute affectsSafety : Boolean = true;
            attribute affectsAntiTheft : Boolean = false;
        }

        part alarmSystem : AntiTheftComponent {
            attribute componentName : String = "Alarm System";
            attribute category : ComponentCategory = ComponentCategory::ANTI_THEFT;
            attribute mandatoryStatus : MandatoryStatus = MandatoryStatus::MANDATORY;
            attribute affectsSafety : Boolean = false;
            attribute affectsAntiTheft : Boolean = true;
        }

        part keylessEntrySystem : AntiTheftComponent {
            attribute componentName : String = "Keyless Entry System";
            attribute category : ComponentCategory = ComponentCategory::ANTI_THEFT;
            attribute mandatoryStatus : MandatoryStatus = MandatoryStatus::MANDATORY;
            attribute affectsSafety : Boolean = false;
            attribute affectsAntiTheft : Boolean = true;
        }
    }

    part def VehicleSafetyCatalogue {
        attribute catalogueName : String = "Vehicle Safety and Anti-Theft Function Catalogue";
        attribute version : String = "1.0";

        part bodyInterior : BodyInterior {
            attribute sectionName : String = "Body and Interior";
        }
    }

    part vehicleCatalogue : VehicleSafetyCatalogue;
}