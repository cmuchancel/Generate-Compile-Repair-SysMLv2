package FuelEconomySystem {
    private import ScalarValues::*;

    enum def DrivingCondition {
        enum urban;
        enum highway;
    }

    part def Vehicle {
        attribute curbWeight : Real;
        attribute urbanFuelEconomy : Real;
        attribute highwayFuelEconomy : Real;
        part load : AdditionalLoad;
        port testPort : FuelTestPort;
    }

    part def AdditionalLoad {
        attribute weight : Real = 1000;
    }

    part def TestScenario {
        attribute condition : DrivingCondition;
        attribute requiredMinimum : Real;
        port vehiclePort : FuelTestPort;
    }

    port def FuelTestPort {
        attribute measuredEconomy : Real;
        attribute requiredEconomy : Real;
    }

    part regulatoryTest : TestScenario [
        part urbanTest : TestScenario {
            attribute condition : DrivingCondition = DrivingCondition::urban;
            attribute requiredMinimum : Real = 25;
        }
        part highwayTest : TestScenario {
            attribute condition : DrivingCondition = DrivingCondition::highway;
            attribute requiredMinimum : Real = 30;
        }
    ]

    part testVehicle : Vehicle [
        attribute curbWeight : Real = 3000;
        attribute urbanFuelEconomy : Real = 27;
        attribute highwayFuelEconomy : Real = 32;
        part load : AdditionalLoad;
        port testPort : FuelTestPort [
            attribute measuredEconomy : Real = 27;
            attribute requiredEconomy : Real = 25;
        ]
    ]

    connect regulatoryTest.urbanTest.vehiclePort to testVehicle.testPort;
    connect regulatoryTest.highwayTest.vehiclePort to testVehicle.testPort;
}