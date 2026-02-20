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
        part alarm : Alarm;
        part keylessEntry : KeylessEntry;
        part safetySecurityManager : SafetySecurityManager;
    }

    part def SafetyFeature {
        attribute featureName : String;
        attribute category : SafetyCategory;
        attribute featureType : FeatureType;
        attribute isMandatory : Boolean;
        attribute description : String;
    }

    part def SecurityFeature {
        attribute featureName : String;
        attribute featureType : FeatureType;
        attribute description : String;
    }

    part def SeatBelt {
        attribute featureName : String = "SeatBelt";
        attribute category : SafetyCategory = SafetyCategory::Mandatory;
        attribute featureType : FeatureType = FeatureType::Safety;
        attribute isMandatory : Boolean = true;
        attribute description : String = "Mandatory seat belt safety feature";
    }

    part def Bumper {
        attribute featureName : String = "Bumper";
        attribute category : SafetyCategory = SafetyCategory::Mandatory;
        attribute featureType : FeatureType = FeatureType::Safety;
        attribute isMandatory : Boolean = true;
        attribute description : String = "Mandatory bumper safety feature";
    }

    part def Airbag {
        attribute featureName : String = "Airbag";
        attribute category : SafetyCategory = SafetyCategory::Optional;
        attribute featureType : FeatureType = FeatureType::Safety;
        attribute isMandatory : Boolean = false;
        attribute description : String = "Optional airbag safety feature";
    }

    part def AntiLockBrakingSystem {
        attribute featureName : String = "AntiLockBrakingSystem";
        attribute category : SafetyCategory = SafetyCategory::Optional;
        attribute featureType : FeatureType = FeatureType::Safety;
        attribute isMandatory : Boolean = false;
        attribute description : String = "Optional anti-lock braking system safety feature";
    }

    part def Alarm {
        attribute featureName : String = "Alarm";
        attribute featureType : FeatureType = FeatureType::Security;
        attribute description : String = "Vehicle alarm security device";
    }

    part def KeylessEntry {
        attribute featureName : String = "KeylessEntry";
        attribute featureType : FeatureType = FeatureType::Security;
        attribute description : String = "Keyless entry security device";
    }

    part def SafetySecurityManager {
        attribute managerName : String = "SafetySecurityManager";
        attribute displayFormat : DisplayFormat;
        attribute filterBySafety : Boolean;
        attribute filterBySecurity : Boolean;
        part mandatorySafetyRegistry : MandatorySafetyRegistry;
        part optionalSafetyRegistry : OptionalSafetyRegistry;
        part securityRegistry : SecurityRegistry;
        part displayEngine : DisplayEngine;
    }

    part def MandatorySafetyRegistry {
        attribute registryName : String = "MandatorySafetyRegistry";
        part seatBeltEntry : SeatBelt;
        part bumperEntry : Bumper;
    }

    part def OptionalSafetyRegistry {
        attribute registryName : String = "OptionalSafetyRegistry";
        part airbagEntry : Airbag;
        part absEntry : AntiLockBrakingSystem;
    }

    part def SecurityRegistry {
        attribute registryName : String = "SecurityRegistry";
        part alarmEntry : Alarm;
        part keylessEntryEntry : KeylessEntry;
    }

    part def DisplayEngine {
        attribute engineName : String = "DisplayEngine";
        attribute supportedFormats : String = "TreeStructure, Table";
        attribute canFilterBySafety : Boolean = true;
        attribute canFilterBySecurity : Boolean = true;
        attribute canDistinguishMandatory : Boolean = true;
    }

    port def FeatureQueryPort {
        attribute queryType : String;
    }

    port def FeatureResponsePort {
        attribute responseFormat : String;
    }

    part def FeatureQueryInterface {
        port queryIn : FeatureQueryPort;
        port responseOut : FeatureResponsePort;
    }

    part vehicleSystem : Vehicle {
        attribute vehicleName = "MainVehicle";
        part seatBelt : SeatBelt;
        part bumper : Bumper;
        part airbag : Airbag;
        part antiLockBrakingSystem : AntiLockBrakingSystem;
        part alarm : Alarm;
        part keylessEntry : KeylessEntry;
        part safetySecurityManager : SafetySecurityManager {
            attribute displayFormat = DisplayFormat::TreeStructure;
            attribute filterBySafety = true;
            attribute filterBySecurity = true;
        }
        part queryInterface : FeatureQueryInterface;
    }
}