package VehicleSafetyComplianceSystem {
    private import ScalarValues::*;

    enum def SafetyRequirementKind {
        mandatory;
        optional;
    }

    part def SeatBelt {
        attribute requirement : SafetyRequirementKind = SafetyRequirementKind::mandatory;
    }
    part def DriversAirbag {
        attribute requirement : SafetyRequirementKind = SafetyRequirementKind::optional;
    }
    part def Bumper {
        attribute requirement : SafetyRequirementKind = SafetyRequirementKind::mandatory;
    }
    part def KeylessEntry {
        attribute requirement : SafetyRequirementKind = SafetyRequirementKind::optional;
    }
    part def Tire;
    part def ABS {
        attribute requirement : SafetyRequirementKind = SafetyRequirementKind::optional;
    }

    part def VehicleInterior {
        part seatBelt1 : SeatBelt;
        part seatBelt2 : SeatBelt;
        part driversAirbag : DriversAirbag;
    }

    part def VehicleBody {
        part bumper : Bumper;
        part keylessEntry : KeylessEntry;
    }

    part def ChassisSystem {
        part tire1 : Tire;
        part tire2 : Tire;
        part tire3 : Tire;
        part tire4 : Tire;
        part abs : ABS;
    }

    part def Vehicle {
        part interior : VehicleInterior;
        part body : VehicleBody;
        part chassis : ChassisSystem;
    }
}