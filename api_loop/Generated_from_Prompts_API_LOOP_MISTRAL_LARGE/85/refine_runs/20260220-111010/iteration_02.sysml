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
        part seatBelts : SeatBelt[2];
        part frontSeats : FrontSeat[2];
        part driverAirbag : DriverAirbag;
        part alarmSystem : AntiTheftAlarm;
    }

    part def BodyAssembly {
        part bodyShell : BodyShell;
        part bumpers : Bumper[2];
        part keylessEntryDevice : KeylessEntry;
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