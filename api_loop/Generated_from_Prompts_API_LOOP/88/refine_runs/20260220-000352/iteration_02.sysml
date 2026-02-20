package VehicleSafetyCompliance {
  private import ScalarValues::*;

  enum def SafetyRequirement {
    required;
    optional;
  }

  part def SeatBelt {
    attribute requirementStatus : SafetyRequirement = SafetyRequirement::required;
  }

  part def Airbag {
    attribute requirementStatus : SafetyRequirement = SafetyRequirement::optional;
  }

  part def Bumper {
    attribute requirementStatus : SafetyRequirement = SafetyRequirement::required;
  }

  part def KeylessEntry {
    attribute requirementStatus : SafetyRequirement = SafetyRequirement::optional;
  }

  part def ABS {
    attribute requirementStatus : SafetyRequirement = SafetyRequirement::optional;
  }

  part def Tire {
  }

  part def VehicleInterior {
    part seatBelt1 : SeatBelt;
    part seatBelt2 : SeatBelt;
    part driverAirbag : Airbag;
  }

  part def VehicleBody {
    part bumper : Bumper;
    part keylessEntrySystem : KeylessEntry;
  }

  part def ChassisSystem {
    attribute tireCount : Integer = 4;
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

  part def SafetyCatalog {
    part seatBelt1 : SeatBelt;
    part seatBelt2 : SeatBelt;
    part driverAirbag : Airbag;
    part bumper : Bumper;
    part keylessEntrySystem : KeylessEntry;
    part abs : ABS;
  }

  part def VehicleSafetyComplianceSystem {
    part vehicle : Vehicle;
    part safetyCatalog : SafetyCatalog;
  }

  part system : VehicleSafetyComplianceSystem;
}