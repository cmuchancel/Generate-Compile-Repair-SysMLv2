package VehicleSafetySecurityManagement {
    private import ScalarValues::*;

    enum def SafetyCategory {
        enum Mandatory;
        enum Optional;
    }

    enum def FeatureType {
        enum Safety;
        enum Security;
    }

    enum def DisplayFormat {
        enum TreeStructure;
        enum Table;
    }

    part def Vehicle {
        attribute vehicleName : String;
        part seatBelt : SeatBelt;
        part bumper : Bumper;
        part airbag : Airbag;
        part antiLockBrakingSystem : AntiLockBrakingSystem;
        part alarmDevice : AlarmDevice;
        part keylessEntry : KeylessEntrySystem;
        part safetySecurityManager : SafetySecurityManager;
    }

    part def SafetyFeature {
        attribute featureName : String;
        attribute category : SafetyCategory;
        attribute featureType : FeatureType;
        attribute isMandatory : Boolean;
    }

    part def SecurityFeature {
        attribute featureName : String;
        attribute featureType : FeatureType;
        attribute isActive : Boolean;
    }

    part def SeatBelt {
        attribute featureName : String = "SeatBelt";
        attribute category : SafetyCategory = SafetyCategory::Mandatory;
        attribute featureType : FeatureType = FeatureType::Safety;
        attribute isMandatory : Boolean = true;
    }

    part def Bumper {
        attribute featureName : String = "Bumper";
        attribute category : SafetyCategory = SafetyCategory::Mandatory;
        attribute featureType : FeatureType = FeatureType::Safety;
        attribute isMandatory : Boolean = true;
    }

    part def Airbag {
        attribute featureName : String = "Airbag";
        attribute category : SafetyCategory = SafetyCategory::Optional;
        attribute featureType : FeatureType = FeatureType::Safety;
        attribute isMandatory : Boolean = false;
    }

    part def AntiLockBrakingSystem {
        attribute featureName : String = "AntiLockBrakingSystem";
        attribute category : SafetyCategory = SafetyCategory::Optional;
        attribute featureType : FeatureType = FeatureType::Safety;
        attribute isMandatory : Boolean = false;
    }

    part def AlarmDevice {
        attribute featureName : String = "AlarmDevice";
        attribute featureType : FeatureType = FeatureType::Security;
        attribute isActive : Boolean = true;
    }

    part def KeylessEntrySystem {
        attribute featureName : String = "KeylessEntrySystem";
        attribute featureType : FeatureType = FeatureType::Security;
        attribute isActive : Boolean = true;
    }

    part def SafetySecurityManager {
        attribute managerName : String = "SafetySecurityManager";
        attribute displayFormat : DisplayFormat;
        attribute filterBySafety : Boolean;
        attribute filterBySecurity : Boolean;
    }

    port def SafetyDataPort {
        attribute portName : String;
    }

    port def SecurityDataPort {
        attribute portName : String;
    }

    part def DisplaySystem {
        attribute systemName : String = "DisplaySystem";
        attribute currentFormat : DisplayFormat;
        port safetyInput : SafetyDataPort;
        port securityInput : SecurityDataPort;
    }

    part def VehicleSystem {
        attribute systemName : String = "VehicleSystem";
        part vehicle : Vehicle;
        part display : DisplaySystem;
        part mandatorySafetyGroup : MandatorySafetyGroup;
        part optionalSafetyGroup : OptionalSafetyGroup;
        part securityGroup : SecurityGroup;
    }

    part def MandatorySafetyGroup {
        attribute groupName : String = "MandatorySafetyFeatures";
        attribute groupCategory : SafetyCategory = SafetyCategory::Mandatory;
        part seatBelt : SeatBelt;
        part bumper : Bumper;
    }

    part def OptionalSafetyGroup {
        attribute groupName : String = "OptionalSafetyFeatures";
        attribute groupCategory : SafetyCategory = SafetyCategory::Optional;
        part airbag : Airbag;
        part antiLockBrakingSystem : AntiLockBrakingSystem;
    }

    part def SecurityGroup {
        attribute groupName : String = "SecurityFeatures";
        attribute groupFeatureType : FeatureType = FeatureType::Security;
        part alarmDevice : AlarmDevice;
        part keylessEntry : KeylessEntrySystem;
    }
}