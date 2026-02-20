package VehiclePerformanceAnalysis {
    private import ScalarValues::*;

    part def Vehicle {
        attribute curbWeight : Real;
        attribute payload : Real;
        attribute wheelDiameter : Real;
        attribute transmissionEfficiency : Real;
        attribute fuelEconomyUrban : Real;
        attribute fuelEconomyHighway : Real;
    }

    part def DrivingCycle {
        attribute time : Real[*];
        attribute position : Real[*];
        attribute speed : Real[*];
    }

    item def EnergyConsumptionTarget {
        attribute fuelConsumptionPer100km : Real;
        attribute milesPerGallon : Real;
    }

    part def PerformanceAnalyzer {
        attribute currentVehicle : Vehicle;
        attribute currentDrivingCycle : DrivingCycle;
        attribute currentTarget : EnergyConsumptionTarget;
        attribute requiredPower : Real[*];
        attribute acceleration : Real[*];
        attribute calculatedFuelEconomy : Real;
        attribute isTargetMet : Boolean;
    }

    part vehicleInstance : Vehicle;
    part driveCycleInstance : DrivingCycle;
    part targetInstance : EnergyConsumptionTarget;
    part analyzerInstance : PerformanceAnalyzer;

    connect analyzerInstance to vehicleInstance;
    connect analyzerInstance to driveCycleInstance;
    connect analyzerInstance to targetInstance;
}