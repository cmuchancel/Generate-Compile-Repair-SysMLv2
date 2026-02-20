package VehiclePerformanceAnalysis {
  private import ScalarValues::*;

  enum def FuelEconomyUnit {
    enum MPG;
    enum L_per_100km;
  }

  item def DrivePoint {
    attribute time : Real;
    attribute position : Real;
    attribute speed : Real;
  }

  item def PerformanceResults {
    attribute powerDemand : Real;
    attribute acceleration : Real;
    attribute speed : Real;
    attribute position : Real;
    attribute fuelEconomy : Real;
  }

  port def DataPort;

  part def Vehicle {
    attribute curbWeight : Real;
    attribute payload : Real;
    attribute wheelDiameter : Real;
    attribute transmissionEfficiency : Real;
    attribute fuelEconomyUrban : Real;
    attribute fuelEconomyHighway : Real;
  }

  part def DriveCycle {
    attribute numberOfPoints : Integer;
    attribute totalTime : Real;
    attribute totalDistance : Real;
    attribute samplePoint : DrivePoint;
    port dataOut : DataPort;
  }

  part def PerformanceAnalyzer {
    attribute results : PerformanceResults;
    port dataIn : DataPort;
    port resultsOut : DataPort;
  }

  part def FuelEconomyCalculator {
    attribute fuelEconomyCycle : Real;
    attribute unit : FuelEconomyUnit;
    port dataIn : DataPort;
    port resultsOut : DataPort;
  }

  part def ComplianceChecker {
    attribute targetFuelEconomy : Real;
    attribute unit : FuelEconomyUnit;
    attribute isCompliant : Boolean;
    port dataIn : DataPort;
  }

  part def VehiclePerformanceSystem {
    part vehicle : Vehicle;
    part driveCycle : DriveCycle;
    part analyzer : PerformanceAnalyzer;
    part fuelEconomy : FuelEconomyCalculator;
    part compliance : ComplianceChecker;
    connect driveCycle.dataOut to analyzer.dataIn;
    connect analyzer.resultsOut to fuelEconomy.dataIn;
    connect fuelEconomy.resultsOut to compliance.dataIn;
  }
}