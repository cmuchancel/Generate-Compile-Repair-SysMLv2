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
        part keylessEntrySystem : KeylessEntrySystem[0..*];
    }

    item def SeatBelt {
        attribute category : FeatureCategory;
        attribute requirement : RequirementType;
    }

    item def Bumper {
        attribute category : FeatureCategory;
        attribute requirement : RequirementType;
    }

    item def Airbag {
        attribute category : FeatureCategory;
        attribute requirement : RequirementType;
    }

    item def AntiLockBrakingSystem {
        attribute category : FeatureCategory;
        attribute requirement : RequirementType;
    }

    item def Alarm {
        attribute category : FeatureCategory;
    }

    item def KeylessEntrySystem {
        attribute category : FeatureCategory;
    }
}