package VehicleSafetyCompliance {
    private import ScalarValues::*;

    enum def SafetyCategory {
        enum MANDATORY;
        enum OPTIONAL;
    }

    part def SafetyFeature {
        attribute category : SafetyCategory;
        attribute isRequired : Boolean;
    }

    part def SeatBelt :> SafetyFeature {
        attribute category : SafetyCategory = SafetyCategory::MANDATORY;
        attribute isRequired : Boolean = true;
    }

    part def DriverAirbag :> SafetyFeature {
        attribute category : SafetyCategory = SafetyCategory::OPTIONAL;
        attribute isRequired : Boolean = false;
    }

    part def Bumper :> SafetyFeature {
        attribute category : SafetyCategory = SafetyCategory::MANDATORY;
        attribute isRequired : Boolean = true;
    }

    part def KeylessEntry :> SafetyFeature {
        attribute category : SafetyCategory = SafetyCategory::OPTIONAL;
        attribute isRequired : Boolean = false;
    }

    part def ABS :> SafetyFeature {
        attribute category : SafetyCategory = SafetyCategory::OPTIONAL;
        attribute isRequired : Boolean = false;
    }

    part def Tire {
        attribute count : Integer;
    }

    part def VehicleInterior {
        part seatBelt1 : SeatBelt;
        part seatBelt2 : SeatBelt;
        part driverAirbag : DriverAirbag;
    }

    part def VehicleBody {
        part bumper : Bumper;
        part keylessEntry : KeylessEntry;
    }

    part def ChassisSystem {
        part tires : Tire {
            attribute count : Integer = 4;
        }
        part abs : ABS;
    }

    part def Vehicle {
        part interior : VehicleInterior;
        part body : VehicleBody;
        part chassis : ChassisSystem;
    }

    part vehicle : Vehicle {
        part interior : VehicleInterior {
            part seatBelt1 : SeatBelt {
                attribute category : SafetyCategory = SafetyCategory::MANDATORY;
                attribute isRequired : Boolean = true;
            }
            part seatBelt2 : SeatBelt {
                attribute category : SafetyCategory = SafetyCategory::MANDATORY;
                attribute isRequired : Boolean = true;
            }
            part driverAirbag : DriverAirbag {
                attribute category : SafetyCategory = SafetyCategory::OPTIONAL;
                attribute isRequired : Boolean = false;
            }
        }
        part body : VehicleBody {
            part bumper : Bumper {
                attribute category : SafetyCategory = SafetyCategory::MANDATORY;
                attribute isRequired : Boolean = true;
            }
            part keylessEntry : KeylessEntry {
                attribute category : SafetyCategory = SafetyCategory::OPTIONAL;
                attribute isRequired : Boolean = false;
            }
        }
        part chassis : ChassisSystem {
            part tires : Tire {
                attribute count : Integer = 4;
            }
            part abs : ABS {
                attribute category : SafetyCategory = SafetyCategory::OPTIONAL;
                attribute isRequired : Boolean = false;
            }
        }
    }
}