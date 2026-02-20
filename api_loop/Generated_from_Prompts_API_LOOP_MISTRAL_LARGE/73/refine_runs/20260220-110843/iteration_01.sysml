package VehicleFuelEconomySimulation {
    private import ScalarValues::*;

    enum def DrivingScenario {
        enum urban;
        enum highway;
    }

    part def Vehicle {
        attribute curbWeight : Real;
        attribute load : Real;
        attribute wheelDiameter : Real;
        attribute transmissionEfficiency : Real;
    }

    part def Engine {
        attribute fuelConsumptionRate : Real;
    }

    part def DrivingTrajectory {
        attribute speedVariation : Real;
        attribute scenario : DrivingScenario;
    }

    part def SimulationResult {
        attribute wheelPower : Real;
        attribute acceleration : Real;
        attribute energyConsumption : Real;
        attribute enginePowerDemand : Real;
        attribute fuelConsumptionLitersPer100km : Real;
        attribute fuelEconomyMpg : Real;
    }

    part def FuelEconomyValidator {
        attribute minimumRequiredMpg : Real = 30.0;
    }

    part simulationSystem {
        part vehicle : Vehicle;
        part engine : Engine;
        part trajectory : DrivingTrajectory;
        part result : SimulationResult;
        part validator : FuelEconomyValidator;

        connect vehicle to result;
        connect engine to result;
        connect trajectory to result;
        connect result to validator;
    }
}