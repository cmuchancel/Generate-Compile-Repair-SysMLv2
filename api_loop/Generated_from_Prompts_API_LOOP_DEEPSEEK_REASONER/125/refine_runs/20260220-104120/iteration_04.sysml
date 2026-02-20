package VehicleSafetySecurity {
    private import ScalarValues::*;

    enum def FeatureCategory {
        enum Safety;
        enum Security;
    }

    enum def SafetyRequirement {
        enum Mandatory;
        enum Optional;
    }

    part def SeatBelt {
        attribute category : FeatureCategory = FeatureCategory::Safety;
        attribute requirement : SafetyRequirement = SafetyRequirement::Mandatory;
    }

    part def Bumper {
        attribute category : FeatureCategory = FeatureCategory::Safety;
        attribute requirement : SafetyRequirement = SafetyRequirement::Mandatory;
    }

    part def Airbag {
        attribute category : FeatureCategory = FeatureCategory::Safety;
        attribute requirement : SafetyRequirement = SafetyRequirement::Optional;
    }

    part def AntiLockBraking {
        attribute category : FeatureCategory = FeatureCategory::Safety;
        attribute requirement : SafetyRequirement = SafetyRequirement::Optional;
    }

    part def Alarm {
        attribute category : FeatureCategory = FeatureCategory::Security;
    }

    part def KeylessEntry {
        attribute category : FeatureCategory = FeatureCategory::Security;
    }

    part def Vehicle {
        part seatBelt : SeatBelt;
        part bumper : Bumper;
        part airbag : Airbag;
        part antiLockBraking : AntiLockBraking;
        part alarm : Alarm;
        part keylessEntry : KeylessEntry;
    }
}