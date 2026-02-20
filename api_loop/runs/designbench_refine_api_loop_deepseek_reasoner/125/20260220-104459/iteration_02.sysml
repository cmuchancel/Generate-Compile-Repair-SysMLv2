package VehicleSafetySecurity {
    private import ScalarValues::*;

    enum def FeatureCategory {
        enum Safety;
        enum Security;
    }

    enum def RequirementType {
        enum Mandatory;
        enum Optional;
    }

    part def Vehicle {
        part seatBelt : SeatBelt;
        part bumper : Bumper;
        part airbag : Airbag[0..*];
        part antiLockBrakingSystem : AntiLockBrakingSystem[0..*];
        part alarm : Alarm[0..*];
        part keylessEntrySystem :lessEntrySystem[0..*];
    }

    part def SeatBelt {
        attribute category : FeatureCategory = FeatureCategory::Safety;
        attribute requirement : RequirementType = RequirementType::Mandatory;
    }

    part def Bumper {
        attribute category : FeatureCategory = FeatureCategory::Safety;
        attribute requirement : RequirementType = RequirementType::Mandatory;
    }

    part def Airbag {
        attribute category : FeatureCategory = FeatureCategory::Safety;
        attribute requirement : RequirementType = RequirementType::Optional;
    }

    part def AntiLockBrakingSystem {
        attribute category : FeatureCategory = FeatureCategory::Safety;
        attribute requirement : RequirementType = RequirementType::Optional;
    }

    part def Alarm {
        attribute category : FeatureCategory = FeatureCategory::Security;
    }

    part def KeylessEntrySystem {
        attribute category : FeatureCategory = FeatureCategory::Security;
    }
}