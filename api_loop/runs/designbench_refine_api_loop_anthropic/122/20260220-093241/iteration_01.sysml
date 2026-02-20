package FuelEconomySystem {
    private import ScalarValues::*;

    enum def DrivingCondition {
        enum Urban;
        enum Highway;
    }

    part def Powertrain {
        attribute engineDisplacement : Real;
        attribute horsepower : Real;
        attribute torque : Real;
    }

    part def Vehicle {
        attribute curbWeight : Real;
        attribute additionalLoad : Real;
        attribute totalWeight : Real;
        attribute urbanFuelEconomy : Real;
        attribute highwayFuelEconomy : Real;
        attribute minUrbanFuelEconomy : Real;
        attribute minHighwayFuelEconomy : Real;
        attribute gallonVolumeCubicInches : Real;
        attribute urbanRequirementMet : Boolean;
        attribute highwayRequirementMet : Boolean;
        part vehiclePowertrain : Powertrain;
    }

    part def FuelEconomyTest {
        attribute testLoad : Real;
        attribute condition : DrivingCondition;
        attribute measuredFuelEconomy : Real;
        attribute minimumRequired : Real;
        attribute requirementMet : Boolean;
    }

    part def FuelEconomySystem {
        part testVehicle : Vehicle;
        part urbanTest : FuelEconomyTest;
        part highwayTest : FuelEconomyTest;
    }

    part fuelEconomySystemInstance : FuelEconomySystem {
        part testVehicle : Vehicle {
            attribute curbWeight : Real = 3500.0;
            attribute additionalLoad : Real = 1000.0;
            attribute totalWeight : Real = 4500.0;
            attribute urbanFuelEconomy : Real = 27.0;
            attribute highwayFuelEconomy : Real = 32.0;
            attribute minUrbanFuelEconomy : Real = 25.0;
            attribute minHighwayFuelEconomy : Real = 30.0;
            attribute gallonVolumeCubicInches : Real = 231.0;
            attribute urbanRequirementMet : Boolean = true;
            attribute highwayRequirementMet : Boolean = true;
            part vehiclePowertrain : Powertrain {
                attribute engineDisplacement : Real = 2.5;
                attribute horsepower : Real = 180.0;
                attribute torque : Real = 175.0;
            }
        }
        part urbanTest : FuelEconomyTest {
            attribute testLoad : Real = 1000.0;
            attribute condition : DrivingCondition = DrivingCondition::Urban;
            attribute measuredFuelEconomy : Real = 27.0;
            attribute minimumRequired : Real = 25.0;
            attribute requirementMet : Boolean = true;
        }
        part highwayTest : FuelEconomyTest {
            attribute testLoad : Real = 1000.0;
            attribute condition : DrivingCondition = DrivingCondition::Highway;
            attribute measuredFuelEconomy : Real = 32.0;
            attribute minimumRequired : Real = 30.0;
            attribute requirementMet : Boolean = true;
        }
    }
}