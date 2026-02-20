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
        attribute featureName : String;
        attribute category : SafetyCategory;
        attribute featureType : FeatureType;
        attribute isMandatory : Boolean;
        attribute description : String;
    }

    part def Bumper {
        attribute featureName : String;
        attribute category : SafetyCategory;
        attribute featureType : FeatureType;
        attribute isMandatory : Boolean;
        attribute description : String;
    }

    part def Airbag {
        attribute featureName : String;
        attribute category : SafetyCategory;
        attribute featureType : FeatureType;
        attribute isMandatory : Boolean;
        attribute description : String;
    }

    part def AntiLockBrakingSystem {
        attribute featureName : String;
        attribute category : SafetyCategory;
        attribute featureType : FeatureType;
        attribute isMandatory : Boolean;
        attribute description : String;
    }

    part def Alarm {
        attribute featureName : String;
        attribute featureType : FeatureType;
        attribute description : String;
    }

    part def KeylessEntry {
        attribute featureName : String;
        attribute featureType : FeatureType;
        attribute description : String;
    }

    part def MandatorySafetyRegistry {
        attribute registryName : String;
        part seatBeltEntry : SeatBelt;
        part bumperEntry : Bumper;
    }

    part def OptionalSafetyRegistry {
        attribute registryName : String;
        part airbagEntry : Airbag;
        part absEntry : AntiLockBrakingSystem;
    }

    part def SecurityRegistry {
        attribute registryName : String;
        part alarmEntry : Alarm;
        part keylessEntryEntry : KeylessEntry;
    }

    part def DisplayEngine {
        attribute engineName : String;
        attribute supportedFormats : String;
        attribute canFilterBySafety : Boolean;
        attribute canFilterBySecurity : Boolean;
        attribute canDistinguishMandatory : Boolean;
    }

    part def SafetySecurityManager {
        attribute managerName : String;
        attribute displayFormat : DisplayFormat;
        attribute filterBySafety : Boolean;
        attribute filterBySecurity : Boolean;
        part mandatorySafetyRegistry : MandatorySafetyRegistry;
        part optionalSafetyRegistry : OptionalSafetyRegistry;
        part securityRegistry : SecurityRegistry;
        part displayEngine : DisplayEngine;
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
        part seatBelt : SeatBelt;
        part bumper : Bumper;
        part airbag : Airbag;
        part antiLockBrakingSystem : AntiLockBrakingSystem;
        part alarm : Alarm;
        part keylessEntry : KeylessEntry;
        part safetySecurityManager : SafetySecurityManager;
        part queryInterface : FeatureQueryInterface;
    }
}