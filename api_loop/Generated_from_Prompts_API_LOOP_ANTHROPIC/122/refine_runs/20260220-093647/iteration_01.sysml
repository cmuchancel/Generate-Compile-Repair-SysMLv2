package FuelEconomySystem {
    private import ScalarValues::*;

    enum def DrivingCondition {
        enum Urban;
        enum Highway;
    }

    part def Powertrain {
        attribute engineDisplacementCubicInches : Real;
        attribute nominalHorsePower : Real;
    }

    part def Load {
        attribute testLoadPounds : Real;
    }

    part def FuelMeasurement {
        attribute gallonCubicInches : Real;
    }

    part def UrbanTest {
        attribute condition : DrivingCondition;
        attribute testLoadPounds : Real;
        attribute measuredMilesPerGallon : Real;
        attribute minimumRequiredMilesPerGallon : Real;
        attribute meetsRequirement : Boolean;
    }

    part def HighwayTest {
        attribute condition : DrivingCondition;
        attribute testLoadPounds : Real;
        attribute measuredMilesPerGallon : Real;
        attribute minimumRequiredMilesPerGallon : Real;
        attribute meetsRequirement : Boolean;
    }

    part def FuelEconomySystem {
        attribute systemName : String;
    }

    part fuelEconomySystem : FuelEconomySystem {
        attribute systemName : String = "Vehicle Fuel Economy Compliance System";

        part powertrain : Powertrain {
            attribute engineDisplacementCubicInches : Real = 183.0;
            attribute nominalHorsePower : Real = 150.0;
        }

        part testLoad : Load {
            attribute testLoadPounds : Real = 1000.0;
        }

        part fuelStandard : FuelMeasurement {
            attribute gallonCubicInches : Real = 231.0;
        }

        part urbanTest : UrbanTest {
            attribute condition : DrivingCondition = DrivingCondition::Urban;
            attribute testLoadPounds : Real = 1000.0;
            attribute measuredMilesPerGallon : Real = 27.5;
            attribute minimumRequiredMilesPerGallon : Real = 25.0;
            attribute meetsRequirement : Boolean = true;
        }

        part highwayTest : HighwayTest {
            attribute condition : DrivingCondition = DrivingCondition::Highway;
            attribute testLoadPounds : Real = 1000.0;
            attribute measuredMilesPerGallon : Real = 32.0;
            attribute minimumRequiredMilesPerGallon : Real = 30.0;
            attribute meetsRequirement : Boolean = true;
        }
    }
}