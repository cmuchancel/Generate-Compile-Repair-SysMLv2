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

    part def SeatBelt {
        attribute category : FeatureCategory;
        attribute requirement : RequirementType;
    }

    part def Bumper {
        attribute category : FeatureCategory;
        attribute requirement : RequirementType;
    }

    part def Airbag {
        attribute category : FeatureCategory;
        attribute requirement : RequirementType;
    }

    part def AntiLockBrakingSystem {
        attribute category : FeatureCategory;
        attribute requirement : RequirementType;
    }

    part def Alarm {
        attribute category : FeatureCategory;
    }

    part def KeylessEntrySystem {
        attribute category : FeatureCategory;
    }

    part def Vehicle {
        part seatBelt : SeatBelt;
        part bumper : Bumper;
        part airbag : Airbag[0..*];
        part antiLockBrakingSystem : AntiLockBrakingSystem[0..*];
        part alarm : Alarm[0..*];
        part keylessEntrySystem : KeylessEntrySystem[0..*];
    }
}