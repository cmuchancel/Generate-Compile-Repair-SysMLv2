package VehicleSafetySecuritySystem {
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
        part alarm : Alarm;
        part keylessEntry : KeylessEntry;
        part safetySecurityManager : SafetySecurityManager;
    }

    part def SafetyFeature {
        attribute featureName : String;
        attribute featureType : FeatureType;
        attribute safetyCategory : SafetyCategory;
        attribute isMandatory : Boolean;
    }

    part def SecurityFeature {
        attribute featureName : String;
        attribute featureType : FeatureType;
        attribute isMandatory : Boolean;
    }

    part def SeatBelt {
        attribute featureName : String = "SeatBelt";
        attribute featureType : FeatureType = FeatureType::Safety;
        attribute safetyCategory : SafetyCategory = SafetyCategory::Mandatory;
        attribute isMandatory : Boolean = true;
    }

    part def Bumper {
        attribute featureName : String = "Bumper";
        attribute featureType : FeatureType = FeatureType::Safety;
        attribute safetyCategory : SafetyCategory = SafetyCategory::Mandatory;
        attribute isMandatory : Boolean = true;
    }

    part def Airbag {
        attribute featureName : String = "Airbag";
        attribute featureType : FeatureType = FeatureType::Safety;
        attribute safetyCategory : SafetyCategory = SafetyCategory::Optional;
        attribute isMandatory : Boolean = false;
    }

    part def AntiLockBrakingSystem {
        attribute featureName : String = "AntiLockBrakingSystem";
        attribute featureType : FeatureType = FeatureType::Safety;
        attribute safetyCategory : SafetyCategory = SafetyCategory::Optional;
        attribute isMandatory : Boolean = false;
    }

    part def Alarm {
        attribute featureName : String = "Alarm";
        attribute featureType : FeatureType = FeatureType::Security;
        attribute isMandatory : Boolean = false;
    }

    part def KeylessEntry {
        attribute featureName : String = "KeylessEntry";
        attribute featureType : FeatureType = FeatureType::Security;
        attribute isMandatory : Boolean = false;
    }

    port def SafetySecurityPort {
        attribute portName : String;
        attribute displayFormat : DisplayFormat;
    }

    part def SafetySecurityManager {
        attribute managerName : String = "SafetySecurityManager";
        attribute activeDisplayFormat : DisplayFormat;
        port filterPort : SafetySecurityPort;
        port displayPort : SafetySecurityPort;
    }

    part def FilteredView {
        attribute viewName : String;
        attribute filterType : FeatureType;
        attribute displayFormat : DisplayFormat;
    }

    part def MandatorySafetyView {
        attribute viewName : String = "MandatorySafetyView";
        attribute filterType : FeatureType = FeatureType::Safety;
        attribute safetyCategory : SafetyCategory = SafetyCategory::Mandatory;
        attribute displayFormat : DisplayFormat = DisplayFormat::TreeStructure;
    }

    part def OptionalSafetyView {
        attribute viewName : String = "OptionalSafetyView";
        attribute filterType : FeatureType = FeatureType::Safety;
        attribute safetyCategory : SafetyCategory = SafetyCategory::Optional;
        attribute displayFormat : DisplayFormat = DisplayFormat::Table;
    }

    part def SecurityView {
        attribute viewName : String = "SecurityView";
        attribute filterType : FeatureType = FeatureType::Security;
        attribute displayFormat : DisplayFormat = DisplayFormat::TreeStructure;
    }

    part vehicleSystem : Vehicle {
        part seatBelt : SeatBelt;
        part bumper : Bumper;
        part airbag : Airbag;
        part antiLockBrakingSystem : AntiLockBrakingSystem;
        part alarm : Alarm;
        part keylessEntry : KeylessEntry;
        part safetySecurityManager : SafetySecurityManager;
    }

    connect vehicleSystem.safetySecurityManager.filterPort to vehicleSystem.safetySecurityManager.displayPort;
}