package FuelEconomySystem {
    private import ScalarValues::*;

    enum def DrivingCondition {
        enum Urban;
        enum Highway;
    }

    item def Fuel {
        attribute volumeGallons : Real;
    }

    item def Load {
        attribute weightPounds : Real;
    }

    part def Powertrain {
        attribute engineDisplacementCubicInches : Real;
        attribute nominalHorsePower : Real;
    }

    part def FuelTank {
        attribute capacityGallons : Real;
        attribute cubicInchesPerUSGallon : Real;
    }

    part def Vehicle {
        attribute baseWeightPounds : Real;
        attribute testLoadPounds : Real;
        attribute totalTestWeightPounds : Real;
        attribute urbanMilesPerGallon : Real;
        attribute highwayMilesPerGallon : Real;
        attribute minUrbanMilesPerGallon : Real;
        attribute minHighwayMilesPerGallon : Real;
        attribute urbanRequirementMet : Boolean;
        attribute highwayRequirementMet : Boolean;
        attribute drivingCondition : DrivingCondition;
        part powertrain : Powertrain;
        part tank : FuelTank;
    }

    part def FuelEconomyTest {
        attribute testLoadPounds : Real;
        attribute gallonCubicInches : Real;
        attribute urbanMPGMeasured : Real;
        attribute highwayMPGMeasured : Real;
        attribute urbanMinRequired : Real;
        attribute highwayMinRequired : Real;
        attribute urbanPassFail : Boolean;
        attribute highwayPassFail : Boolean;
        part testVehicle : Vehicle;
    }

    part def FuelEconomySystemDef {
        part urbanTest : FuelEconomyTest;
        part highwayTest : FuelEconomyTest;
    }

    part fuelEconomySystem : FuelEconomySystemDef {
        part urbanTest : FuelEconomyTest {
            attribute :>> testLoadPounds = 1000.0;
            attribute :>> gallonCubicInches = 231.0;
            attribute :>> urbanMinRequired = 25.0;
            attribute :>> highwayMinRequired = 30.0;
            attribute :>> urbanMPGMeasured = 27.5;
            attribute :>> highwayMPGMeasured = 32.0;
            attribute :>> urbanPassFail = true;
            attribute :>> highwayPassFail = true;
            part testVehicle : Vehicle {
                attribute :>> baseWeightPounds = 3500.0;
                attribute :>> testLoadPounds = 1000.0;
                attribute :>> totalTestWeightPounds = 4500.0;
                attribute :>> urbanMilesPerGallon = 27.5;
                attribute :>> highwayMilesPerGallon = 32.0;
                attribute :>> minUrbanMilesPerGallon = 25.0;
                attribute :>> minHighwayMilesPerGallon = 30.0;
                attribute :>> urbanRequirementMet = true;
                attribute :>> highwayRequirementMet = true;
                attribute :>> drivingCondition = DrivingCondition::Urban;
                part powertrain : Powertrain {
                    attribute :>> engineDisplacementCubicInches = 183.0;
                    attribute :>> nominalHorsePower = 150.0;
                }
                part tank : FuelTank {
                    attribute :>> capacityGallons = 15.0;
                    attribute :>> cubicInchesPerUSGallon = 231.0;
                }
            }
        }

        part highwayTest : FuelEconomyTest {
            attribute :>> testLoadPounds = 1000.0;
            attribute :>> gallonCubicInches = 231.0;
            attribute :>> urbanMinRequired = 25.0;
            attribute :>> highwayMinRequired = 30.0;
            attribute :>> urbanMPGMeasured = 27.5;
            attribute :>> highwayMPGMeasured = 32.0;
            attribute :>> urbanPassFail = true;
            attribute :>> highwayPassFail = true;
            part testVehicle : Vehicle {
                attribute :>> baseWeightPounds = 3500.0;
                attribute :>> testLoadPounds = 1000.0;
                attribute :>> totalTestWeightPounds = 4500.0;
                attribute :>> urbanMilesPerGallon = 27.5;
                attribute :>> highwayMilesPerGallon = 32.0;
                attribute :>> minUrbanMilesPerGallon = 25.0;
                attribute :>> minHighwayMilesPerGallon = 30.0;
                attribute :>> urbanRequirementMet = true;
                attribute :>> highwayRequirementMet = true;
                attribute :>> drivingCondition = DrivingCondition::Highway;
                part powertrain : Powertrain {
                    attribute :>> engineDisplacementCubicInches = 183.0;
                    attribute :>> nominalHorsePower = 150.0;
                }
                part tank : FuelTank {
                    attribute :>> capacityGallons = 15.0;
                    attribute :>> cubicInchesPerUSGallon = 231.0;
                }
            }
        }
    }
}