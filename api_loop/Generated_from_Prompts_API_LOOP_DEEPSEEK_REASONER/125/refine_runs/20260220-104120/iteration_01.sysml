package VehicleSafetySecurity {
    private import ScalarValues::*;

    enum def FeatureCategory {
        enum Safety,
        enum Security
    }

    enum def SafetyRequirement {
        enum Mandatory,
        enum Optional
    }

    part def VehicleComponent {
        attribute category : FeatureCategory;
    }

    part def SafetyComponent {
        attribute requirement : SafetyRequirement;
    }

    part def SecurityComponent {
    }

    part def SeatBelt {
        part : SafetyComponent;
        attribute isPresent : Boolean = true;
    }

    part def Bumper {
        part : SafetyComponent;
        attribute isPresent : Boolean = true;
    }

    part def Airbag {
        part : SafetyComponent;
        attribute isPresent : Boolean = true;
    }

    part def AntiLockBraking {
        part : SafetyComponent;
        attribute isPresent : Boolean = true;
    }

    part def Alarm {
        part : SecurityComponent;
        attribute isPresent : Boolean = true;
    }

    part def KeylessEntry {
        part : SecurityComponent;
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