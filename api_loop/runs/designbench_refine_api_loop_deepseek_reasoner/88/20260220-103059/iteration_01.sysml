package VehicleSafetyComplianceSystem {
    private import ScalarValues::*;
    
    enum def SafetyRequirementKind {
        mandatory : [1];
        optional : [1];
    }
    
    part def SafetyFeature {
        attribute requirement : SafetyRequirementKind;
    }
    
    part def SeatBelt :> SafetyFeature;
    part def DriversAirbag :> SafetyFeature;
    part def Bumper :> SafetyFeature;
    part def KeylessEntry :> SafetyFeature;
    part def Tire;
    part def ABS :> SafetyFeature;
    
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
    
    attribute VehicleInterior::seatBelt1.requirement = SafetyRequirementKind::mandatory;
    attribute VehicleInterior::seatBelt2.requirement = SafetyRequirementKind::mandatory;
    attribute VehicleInterior::driversAirbag.requirement = SafetyRequirementKind::optional;
    attribute VehicleBody::bumper.requirement = SafetyRequirementKind::mandatory;
    attribute VehicleBody::keylessEntry.requirement = SafetyRequirementKind::optional;
    attribute ChassisSystem::abs.requirement = SafetyRequirementKind::optional;
}