package VehicleSafetySecurity {

    enum def FeatureCategory {
        enum Safety;
        enum Security;
    }

    enum def SafetyRequirement {
        enum Mandatory;
        enum Optional;
    }

    part def SeatBelt {
        attribute category : FeatureCategory;
        attribute requirement : SafetyRequirement;
    }

    part def Bumper {
        attribute category : FeatureCategory;
        attribute requirement : SafetyRequirement;
    }

    part def Airbag {
        attribute category : FeatureCategory;
        attribute requirement : SafetyRequirement;
    }

    part def AntiLockBraking {
        attribute category : FeatureCategory;
        attribute requirement : SafetyRequirement;
    }

    part def Alarm {
        attribute category : FeatureCategory;
    }

    part def KeylessEntry {
        attribute category : FeatureCategory;
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