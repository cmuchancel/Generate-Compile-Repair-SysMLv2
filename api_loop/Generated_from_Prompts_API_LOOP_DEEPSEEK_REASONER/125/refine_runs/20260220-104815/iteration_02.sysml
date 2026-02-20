package VehicleSafetySecurity {
    private import ScalarValues::*;

    enum def FeatureCategory {
        enum SafetyMandatory;
        enum SafetyOptional;
        enum Security;
    }

    part def VehicleComponent {
        attribute category : FeatureCategory;
    }

    part def SeatBelt :> VehicleComponent {
        attribute category : FeatureCategory = SafetyMandatory;
    }

    part def Bumper :> VehicleComponent {
        attribute category : FeatureCategory = SafetyMandatory;
    }

    part def Airbag :> VehicleComponent {
        attribute category : FeatureCategory = SafetyOptional;
    }

    part def AntiLockBrakingSystem :> VehicleComponent {
        attribute category : FeatureCategory = SafetyOptional;
    }

    part def AlarmDevice :> VehicleComponent {
        attribute category : FeatureCategory = Security;
    }

    part def KeylessEntrySystem :> VehicleComponent {
        attribute category : FeatureCategory = Security;
    }

    part def Vehicle {
        part seatBelt : SeatBelt;
        part bumper : Bumper;
        part airbag : Airbag;
        part abs : AntiLockBrakingSystem;
        part alarm : AlarmDevice;
        part keylessEntry : KeylessEntrySystem;
    }
}