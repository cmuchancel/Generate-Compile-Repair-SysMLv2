package VehicleSafetySecurityCompliance {
    private import ScalarValues::*;

    enum def ComplianceCategory {
        enum SAFETY;
        enum SECURITY;
        enum SAFETY_AND_SECURITY;
    }

    enum def MandatoryStatus {
        enum MANDATORY;
        enum OPTIONAL;
    }

    part def SeatBelt {
        attribute category : ComplianceCategory = ComplianceCategory::SAFETY;
        attribute mandatory : MandatoryStatus = MandatoryStatus::MANDATORY;
        attribute quantity : Integer = 2;
    }

    part def Airbag {
        attribute category : ComplianceCategory = ComplianceCategory::SAFETY;
        attribute mandatory : MandatoryStatus = MandatoryStatus::OPTIONAL;
        attribute quantity : Integer = 1;
    }

    part def AntiTheftAlarm {
        attribute category : ComplianceCategory = ComplianceCategory::SECURITY;
        attribute mandatory : MandatoryStatus = MandatoryStatus::MANDATORY;
        attribute quantity : Integer = 1;
    }

    part def Bumper {
        attribute category : ComplianceCategory = ComplianceCategory::SAFETY;
        attribute mandatory : MandatoryStatus = MandatoryStatus::MANDATORY;
        attribute quantity : Integer = 1;
    }

    part def KeylessEntry {
        attribute category : ComplianceCategory = ComplianceCategory::SECURITY;
        attribute mandatory : MandatoryStatus = MandatoryStatus::MANDATORY;
        attribute quantity : Integer = 1;
    }

    part def Wheel {
        attribute category : ComplianceCategory = ComplianceCategory::SAFETY;
        attribute mandatory : MandatoryStatus = MandatoryStatus::MANDATORY;
        attribute quantity : Integer = 2;
    }

    part def AntiLockBrakingSystem {
        attribute category : ComplianceCategory = ComplianceCategory::SAFETY;
        attribute mandatory : MandatoryStatus = MandatoryStatus::OPTIONAL;
        attribute quantity : Integer = 2;
    }

    part def VehicleInterior {
        part seatBelt1 : SeatBelt;
        part seatBelt2 : SeatBelt;
        part airbag : Airbag;
        part antiTheftAlarm : AntiTheftAlarm;
    }

    part def VehicleBody {
        part bumper : Bumper;
        part keylessEntry : KeylessEntry;
    }

    part def Chassis {
        part wheel1 : Wheel;
        part wheel2 : Wheel;
        part abs1 : AntiLockBrakingSystem;
        part abs2 : AntiLockBrakingSystem;
    }

    part def Vehicle {
        attribute complianceStatus : String = "COMPLIANT";
        attribute systemClassification : ComplianceCategory = ComplianceCategory::SAFETY_AND_SECURITY;

        part interior : VehicleInterior;
        part body : VehicleBody;
        part chassis : Chassis;
    }

    part vehicle : Vehicle {
        part interior : VehicleInterior {
            part seatBelt1 : SeatBelt;
            part seatBelt2 : SeatBelt;
            part airbag : Airbag;
            part antiTheftAlarm : AntiTheftAlarm;
        }
        part body : VehicleBody {
            part bumper : Bumper;
            part keylessEntry : KeylessEntry;
        }
        part chassis : Chassis {
            part wheel1 : Wheel;
            part wheel2 : Wheel;
            part abs1 : AntiLockBrakingSystem;
            part abs2 : AntiLockBrakingSystem;
        }
    }
}