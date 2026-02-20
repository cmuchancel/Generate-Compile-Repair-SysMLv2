package VehicleSafetySecuritySystem {
    private import ScalarValues::*;

    enum def SafetyCategory {
        enum mandatory;
        enum optional;
    }

    enum def FeatureType {
        enum safety;
        enum security;
    }

    part def Vehicle {
        attribute safetyFeatures : SafetyFeature[*];
        attribute securityFeatures : SecurityFeature[*];
    }

    part def SafetyFeature {
        attribute name : String;
        attribute category : SafetyCategory;
        attribute isMandatory : Boolean;
    }

    part def SecurityFeature {
        attribute name : String;
    }

    part def SeatBelt :> SafetyFeature {
        attribute name : String = "Seat Belt";
        attribute category : SafetyCategory = SafetyCategory::mandatory;
        attribute isMandatory : Boolean = true;
    }

    part def Bumper :> SafetyFeature {
        attribute name : String = "Bumper";
        attribute category : SafetyCategory = SafetyCategory::mandatory;
        attribute isMandatory : Boolean = true;
    }

    part def Airbag :> SafetyFeature {
        attribute name : String = "Airbag";
        attribute category : SafetyCategory = SafetyCategory::optional;
        attribute isMandatory : Boolean = false;
    }

    part def AntiLockBrakingSystem :> SafetyFeature {
        attribute name : String = "Anti-Lock Braking System";
        attribute category : SafetyCategory = SafetyCategory::optional;
        attribute isMandatory : Boolean = false;
    }

    part def AlarmDevice :> SecurityFeature {
        attribute name : String = "Alarm Device";
    }

    part def KeylessEntrySystem :> SecurityFeature {
        attribute name : String = "Keyless Entry System";
    }

    part vehicle : Vehicle {
        part seatBelt : SeatBelt;
        part bumper : Bumper;
        part airbag : Airbag;
        part abs : AntiLockBrakingSystem;
        part alarm : AlarmDevice;
        part keylessEntry : KeylessEntrySystem;

        attribute safetyFeatures : SafetyFeature[*] = (seatBelt, bumper, airbag, abs);
        attribute securityFeatures : SecurityFeature[*] = (alarm, keylessEntry);
    }
}