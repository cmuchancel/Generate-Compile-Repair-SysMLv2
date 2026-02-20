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

    part def VehicleComponent {
        attribute category : FeatureCategory;
    }

    part def SafetyComponent :> VehicleComponent {
        attribute requirement : SafetyRequirement;
    }

    part def SecurityComponent :> VehicleComponent {
    }

    part def SeatBelt :> SafetyComponent {
    }

    part def Bumper :> SafetyComponent {
    }

    part def Airbag :> SafetyComponent {
    }

    part def AntiLockBraking :> SafetyComponent {
    }

    part def Alarm :> SecurityComponent {
    }

    part def KeylessEntry :> SecurityComponent {
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