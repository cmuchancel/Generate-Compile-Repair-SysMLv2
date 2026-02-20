package VehicleSafetySecuritySystem {
    private import ScalarValues::*;

    enum def SystemCategory {
        enum Safety;
        enum Security;
    }

    part def AlarmSystem {
        attribute category : SystemCategory = SystemCategory::Security;
        attribute isInstalled : Boolean = true;
    }

    part def SeatBelt {
        attribute category : SystemCategory = SystemCategory::Safety;
        attribute isInstalled : Boolean = true;
    }

    part def FrontRowSeat {
        attribute category : SystemCategory = SystemCategory::Safety;
        attribute quantity : Integer = 2;
    }

    part def DriverAirbag {
        attribute category : SystemCategory = SystemCategory::Safety;
        attribute isInstalled : Boolean = true;
    }

    part def Interior {
        part alarm : AlarmSystem;
        part seatBelt : SeatBelt;
        part frontSeat : FrontRowSeat;
        part airbag : DriverAirbag;
    }

    part def VehicleBody {
        attribute material : String = "Steel";
    }

    part def Bumper {
        attribute category : SystemCategory = SystemCategory::Safety;
        attribute isInstalled : Boolean = true;
    }

    part def KeylessEntrySystem {
        attribute category : SystemCategory = SystemCategory::Security;
        attribute isInstalled : Boolean = true;
    }

    part def BodyAssembly {
        part body : VehicleBody;
        part bumper : Bumper;
        part keylessEntry : KeylessEntrySystem;
    }

    part def VehicleSafetySecurityComponent {
        part interior : Interior;
        part bodyAssembly : BodyAssembly;

        attribute hasSafetyFeatures : Boolean = true;
        attribute hasSecurityFeatures : Boolean = true;
    }

    part vehicleSystem : VehicleSafetySecurityComponent {
        part interior : Interior {
            part alarm : AlarmSystem;
            part seatBelt : SeatBelt;
            part frontSeat : FrontRowSeat;
            part airbag : DriverAirbag;
        }
        part bodyAssembly : BodyAssembly {
            part body : VehicleBody;
            part bumper : Bumper;
            part keylessEntry : KeylessEntrySystem;
        }
    }
}