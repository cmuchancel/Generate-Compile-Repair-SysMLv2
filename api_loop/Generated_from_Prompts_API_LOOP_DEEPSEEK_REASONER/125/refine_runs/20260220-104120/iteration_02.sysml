package VehicleSafetySecurity {
    private import ScalarValues::*;

    enum def FeatureCategory {
        enumeration Safety;
        enumeration Security;
    }

    enum def SafetyRequirement {
        enumeration Mandatory;
        enumeration Optional;
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
        attribute isPresent : Boolean = true;
    }

    part def Bumper :> SafetyComponent {
        attribute isPresent : Boolean = true;
    }

    part def Airbag :> SafetyComponent {
        attribute isPresent : Boolean = true;
    }

    part def AntiLockBraking :> SafetyComponent {
        attribute isPresent : Boolean = true;
    }

    part def Alarm :> SecurityComponent {
        attribute isPresent : Boolean = true;
    }

    part def KeylessEntry :> SecurityComponent {
        attribute isPresent : Boolean = true;
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