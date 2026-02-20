package VehicleSafetyComplianceSystem {
    private import ScalarValues::*;

    enum def SafetyRequirementKind {
        mandatory;
        optional;
    }

    part def SeatBelt;
    part def DriversAirbag;
    part def Bumper;
    part def KeylessEntry;
    part def Tire;
    part def ABS;

    part def VehicleInterior {
        part seatBelt1 : SeatBelt;
        part seatBelt2 : SeatBelt;
        part driversAirbag : DriversAirbag;
        attribute seatBeltRequirement : SafetyRequirementKind = SafetyRequirementKind::mandatory;
        attribute airbagRequirement : SafetyRequirementKind = SafetyRequirementKind::optional;
    }

    part def VehicleBody {
        part bumper : Bumper;
        part keylessEntry : KeylessEntry;
        attribute bumperRequirement : SafetyRequirementKind = SafetyRequirementKind::mandatory;
        attribute keylessRequirement : SafetyRequirementKind = SafetyRequirementKind::optional;
    }

    part def ChassisSystem {
        part tire1 : Tire;
        part tire2 : Tire;
        part tire3 : Tire;
        part tire4 : Tire;
        part abs : ABS;
        attribute absRequirement : SafetyRequirementKind = SafetyRequirementKind::optional;
    }

    part def Vehicle {
        part interior : VehicleInterior;
        part body : VehicleBody;
        part chassis : ChassisSystem;
    }
}