package VehicleFuelEconomy {
    private import ScalarValues::*;

    enum def DrivingCondition {
        enum urban;
        enum highway;
    }

    part def Vehicle {
        attribute baseFuelEconomyUrban : Real;
        attribute baseFuelEconomyHighway : Real;
        attribute additionalLoad : Real = 1000.0;
        attribute measuredFuelEconomyUrban : Real;
        attribute measuredFuelEconomyHighway : Real;
    }

    part def Powertrain {
        attribute efficiency : Real;
    }

    part def TestScenario {
        attribute condition : DrivingCondition;
        attribute minFuelEconomy : Real;
    }

    part vehicle : Vehicle;
    part powertrain : Powertrain;
    part urbanTest : TestScenario {
        attribute condition : DrivingCondition = DrivingCondition::urban;
        attribute minFuelEconomy : Real = 25.0;
    }
    part highwayTest : TestScenario {
        attribute condition : DrivingCondition = DrivingCondition::highway;
        attribute minFuelEconomy : Real = 30.0;
    }
}