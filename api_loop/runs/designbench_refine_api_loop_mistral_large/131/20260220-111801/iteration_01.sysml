package VehicleSafetyAndSecurity {
    private import ScalarValues::*;

    enum def SafetyFeature {
        enum seatBelt;
        enum airbag;
        enum bumper;
    }

    enum def SecurityFeature {
        enum alarmSystem;
        enum keylessEntry;
    }

    part def VehicleInterior {
        attribute alarmSystem : Boolean = true;
        attribute seatBelts : Boolean = true;
        attribute frontRowSeats : Boolean = true;
        attribute driversAirbag : Boolean = true;
    }

    part def VehicleBodyAssembly {
        attribute vehicleBody : Boolean = true;
        attribute bumper : Boolean = true;
        attribute keylessEntrySystem : Boolean = true;
    }

    part def VehicleSafetyAndSecuritySystem {
        part interior : VehicleInterior;
        part bodyAssembly : VehicleBodyAssembly;
    }

    part vehicleSystem : VehicleSafetyAndSecuritySystem;
}