package VehicleFeatureAnnotationSystem {
    private import ScalarValues::*;

    enum def SafetyCategory {
        enum SAFETY;
        enum SECURITY;
        enum PROTECTION;
        enum CONVENIENCE;
    }

    part def SeatBelt {
        attribute featureCategory : SafetyCategory = SafetyCategory::SAFETY;
        attribute description : String = "Restraint system for occupant safety";
        attribute isActive : Boolean = true;
    }

    part def DriverAirbag {
        attribute featureCategory : SafetyCategory = SafetyCategory::SAFETY;
        attribute description : String = "Driver airbag for collision protection";
        attribute isActive : Boolean = true;
    }

    part def FrontSeat {
        attribute description : String = "Front seat assembly";
        attribute position : String = "Front";
    }

    part def AlarmSystem {
        attribute featureCategory : SafetyCategory = SafetyCategory::SECURITY;
        attribute description : String = "Anti-theft alarm system";
        attribute isActive : Boolean = true;
    }

    part def BodyShell {
        attribute description : String = "Vehicle body shell structure";
        attribute material : String = "Steel";
    }

    part def Bumper {
        attribute featureCategory : SafetyCategory = SafetyCategory::PROTECTION;
        attribute description : String = "Impact absorbing bumper";
        attribute position : String = "Front and Rear";
    }

    part def KeylessEntryDevice {
        attribute featureCategory : SafetyCategory = SafetyCategory::CONVENIENCE;
        attribute description : String = "Keyless entry system for security and convenience";
        attribute isActive : Boolean = true;
    }

    part def InteriorAssembly {
        part seatBeltDriver : SeatBelt;
        part seatBeltPassenger : SeatBelt;
        part frontSeatDriver : FrontSeat;
        part frontSeatPassenger : FrontSeat;
        part driverAirbag : DriverAirbag;
        part alarmSystem : AlarmSystem;
    }

    part def BodyAssembly {
        part bodyShell : BodyShell;
        part frontBumper : Bumper;
        part rearBumper : Bumper;
        part keylessEntryDevice : KeylessEntryDevice;
    }

    part def Vehicle {
        attribute vehicleName : String = "Complete Vehicle Model";
        attribute modelYear : String = "2024";

        part interior : InteriorAssembly;
        part body : BodyAssembly;
    }

    part vehicle : Vehicle;
}