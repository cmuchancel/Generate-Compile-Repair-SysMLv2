package VehicleFeatureAnnotationSystem {
    private import ScalarValues::*;

    enum def SafetyFeatureType {
        enum SeatBelt;
        enum DriverAirbag;
        enum Bumper;
    }

    enum def SecurityFeatureType {
        enum AntiTheftAlarm;
        enum KeylessEntry;
    }

    part def Vehicle {
        part interior : InteriorAssembly;
        part body : BodyAssembly;
    }

    part def InteriorAssembly {
        part seatBelts : SeatBelt[2] = SeatBelt;
        part frontSeats : FrontSeat[2] = FrontSeat;
        part driverAirbag : DriverAirbag = DriverAirbag;
        part alarmSystem : AntiTheftAlarm = AntiTheftAlarm;
    }

    part def BodyAssembly {
        part bodyShell : BodyShell = BodyShell;
        part bumpers : Bumper[2] = Bumper;
        part keylessEntryDevice : KeylessEntry = KeylessEntry;
    }

    part def SeatBelt {
        attribute safetyType : SafetyFeatureType = SafetyFeatureType::SeatBelt;
    }

    part def FrontSeat;

    part def DriverAirbag {
        attribute safetyType : SafetyFeatureType = SafetyFeatureType::DriverAirbag;
    }

    part def AntiTheftAlarm {
        attribute securityType : SecurityFeatureType = SecurityFeatureType::AntiTheftAlarm;
    }

    part def BodyShell;

    part def Bumper {
        attribute safetyType : SafetyFeatureType = SafetyFeatureType::Bumper;
    }

    part def KeylessEntry {
        attribute securityType : SecurityFeatureType = SecurityFeatureType::KeylessEntry;
    }

    part vehicle : Vehicle;
}