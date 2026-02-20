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
    part urbanTest : TestScenario;
    part highwayTest : TestScenario;

    attribute urbanMin : Real = 25.0;
    attribute highwayMin : Real = 30.0;

    urbanTest.condition = DrivingCondition::urban;
    urbanTest.minFuelEconomy = urbanMin;

    highwayTest.condition = DrivingCondition::highway;
    highwayTest.minFuelEconomy = highwayMin;
}