package FuelEconomySystem {
    private import ScalarValues::*;

    enum def DrivingCondition {
        enum Urban;
        enum Highway;
    }

    item def Fuel {
        attribute volumePerGallon : Real = 231.0;
    }

    part def Powertrain {
        attribute engineDisplacement : Real;
        attribute horsepower : Real;
    }

    part def Vehicle {
        attribute curb_weight : Real;
        attribute test_load : Real = 1000.0;
        attribute total_test_weight : Real;
        attribute urban_fuel_economy : Real;
        attribute highway_fuel_economy : Real;
        attribute min_urban_mpg : Real = 25.0;
        attribute min_highway_mpg : Real = 30.0;
        attribute urban_meets_requirement : Boolean;
        attribute highway_meets_requirement : Boolean;
        part powertrain : Powertrain;
    }

    part def FuelEconomyTest {
        attribute measured_mpg : Real;
        attribute additional_load : Real = 1000.0;
        attribute gallon_cubic_inches : Real = 231.0;
        attribute passed : Boolean;
    }

    part def UrbanTest :> FuelEconomyTest {
        attribute minimum_required_mpg : Real = 25.0;
    }

    part def HighwayTest :> FuelEconomyTest {
        attribute minimum_required_mpg : Real = 30.0;
    }

    part def FuelEconomySystemDef {
        part vehicle : Vehicle;
        part urbanTest : UrbanTest;
        part highwayTest : HighwayTest;
    }

    part fuelEconomySystem : FuelEconomySystemDef {
        part vehicle : Vehicle {
            attribute redefines curb_weight = 3500.0;
            attribute redefines total_test_weight = 4500.0;
            attribute redefines urban_fuel_economy = 27.0;
            attribute redefines highway_fuel_economy = 32.0;
            attribute redefines urban_meets_requirement = true;
            attribute redefines highway_meets_requirement = true;

            part powertrain : Powertrain {
                attribute redefines engineDisplacement = 2.0;
                attribute redefines horsepower = 150.0;
            }
        }

        part urbanTest : UrbanTest {
            attribute redefines measured_mpg = 27.0;
            attribute redefines passed = true;
        }

        part highwayTest : HighwayTest {
            attribute redefines measured_mpg = 32.0;
            attribute redefines passed = true;
        }
    }
}