package VehiclePerformanceAnalysis {
    private import ScalarValues::*;

    enum def DriveCycleType {
        enum Urban;
        enum Highway;
        enum Custom;
    }

    part def Vehicle {
        attribute curbWeight : Real;
        attribute payload : Real;
        attribute wheelDiameter : Real;
        attribute transmissionEfficiency : Real;
        attribute urbanFuelEconomy : Real;
        attribute highwayFuelEconomy : Real;
    }

    part def DriveCycle {
        attribute cycleType : DriveCycleType;
        attribute timeStamps : Real[*];
        attribute positions : Real[*];
        attribute speeds : Real[*];
    }

    part def PerformanceAnalyzer {
        attribute targetFuelEconomy : Real;
        attribute calculatedPowerDemand : Real;
        attribute calculatedAcceleration : Real;
        attribute calculatedSpeed : Real;
        attribute calculatedPosition : Real;
        attribute totalFuelConsumption : Real;
        attribute fuelEconomyResult : Real;
        attribute complianceStatus : Boolean;
    }

    part def SystemContext {
        part vehicle : Vehicle;
        part driveCycle : DriveCycle;
        part analyzer : PerformanceAnalyzer;
    }

    item def PowerDemand;
    item def Acceleration;
    item def Speed;
    item def Position;
    item def FuelConsumption;

    port def VehiclePort {
        out powerDemand : PowerDemand;
        out acceleration : Acceleration;
        out speed : Speed;
        out position : Position;
    }

    port def DriveCyclePort {
        in powerDemand : PowerDemand;
        in acceleration : Acceleration;
        in speed : Speed;
        in position : Position;
    }

    port def AnalyzerPort {
        in powerDemand : PowerDemand;
        in acceleration : Acceleration;
        in speed : Speed;
        in position : Position;
        out fuelConsumption : FuelConsumption;
    }

    part system : SystemContext {
        part vehicle : Vehicle {
            port vehiclePort : VehiclePort;
        }

        part driveCycle : DriveCycle {
            port driveCyclePort : DriveCyclePort;
        }

        part analyzer : PerformanceAnalyzer {
            port analyzerPort : AnalyzerPort;
        }

        connect vehicle.vehiclePort to driveCycle.driveCyclePort;
        connect driveCycle.driveCyclePort to analyzer.analyzerPort;
    }
}