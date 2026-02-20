package VehicleFuelEconomySimulation {
    private import ScalarValues::*;

    enum def DrivingScenarioType {
        enum Urban;
        enum Highway;
    }

    item def DrivingTrajectory {
        attribute speedProfile : Real[1];
        attribute distance : Real;
    }

    part def Vehicle {
        port drivingCondition : DrivingTrajectory;
        port fuelConsumption : Real;

        attribute curbWeight : Real;
        attribute load : Real;
        attribute wheelDiameter : Real;
        attribute transmissionEfficiency : Real;
        attribute currentSpeed : Real;
        attribute acceleration : Real;
        attribute wheelPower : Real;
        attribute enginePower : Real;
        attribute fuelEconomyMPG : Real;
        attribute fuelConsumptionLPer100km : Real;
    }

    part def SimulationController {
        port scenario : DrivingTrajectory;
        port consumption : Real;

        attribute scenarioType : DrivingScenarioType;
        attribute requirementMPG : Real = 30;
        attribute requirementMet : Boolean;
    }

    part def VehicleFuelEconomySystem {
        part vehicle : Vehicle;
        part controller : SimulationController;

        connect controller.scenario to vehicle.drivingCondition;
        connect vehicle.fuelConsumption to controller.consumption;
    }
}