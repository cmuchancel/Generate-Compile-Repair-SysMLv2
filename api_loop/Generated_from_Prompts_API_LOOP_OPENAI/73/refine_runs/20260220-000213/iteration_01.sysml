package VehicleFuelEconomySimulation {
  private import ScalarValues::*;

  enum def DrivingScenarioType {
    enum urban;
    enum highway;
  }

  item def VehicleParamData {
    attribute curbWeight : Real;
    attribute load : Real;
    attribute wheelDiameter : Real;
    attribute transmissionEfficiency : Real;
    attribute frontalArea : Real;
    attribute dragCoefficient : Real;
  }

  item def DrivingScenarioData {
    attribute scenarioType : DrivingScenarioType;
    attribute distanceKm : Real;
    attribute averageSpeedKph : Real;
    attribute speedVariation : Real;
  }

  item def DynamicsData {
    attribute requiredWheelPower : Real;
    attribute acceleration : Real;
    attribute energyConsumption : Real;
  }

  item def EngineDemandData {
    attribute enginePowerDemand : Real;
  }

  item def FuelEconomyData {
    attribute fuelConsumedLiters : Real;
    attribute fuelEconomyMPG : Real;
    attribute fuelEconomyLper100km : Real;
    attribute meetsRequirement : Boolean;
  }

  port def VehicleParamPort {
    attribute data : VehicleParamData;
  }

  port def ScenarioPort {
    attribute data : DrivingScenarioData;
  }

  port def DynamicsPort {
    attribute data : DynamicsData;
  }

  port def EngineDemandPort {
    attribute data : EngineDemandData;
  }

  port def FuelEconomyPort {
    attribute data : FuelEconomyData;
  }

  part def VehicleParameters {
    attribute curbWeight : Real;
    attribute load : Real;
    attribute wheelDiameter : Real;
    attribute transmissionEfficiency : Real;
    attribute frontalArea : Real;
    attribute dragCoefficient : Real;
    port paramsOut : VehicleParamPort;
  }

  part def DrivingScenario {
    attribute scenarioType : DrivingScenarioType;
    attribute distanceKm : Real;
    attribute averageSpeedKph : Real;
    attribute speedVariation : Real;
    port scenarioOut : ScenarioPort;
  }

  part def DynamicsAnalysis {
    attribute requiredWheelPower : Real;
    attribute acceleration : Real;
    attribute energyConsumption : Real;
    port paramsIn : VehicleParamPort;
    port scenarioIn : ScenarioPort;
    port dynamicsOut : DynamicsPort;
  }

  part def EngineDemandAnalysis {
    attribute enginePowerDemand : Real;
    port dynamicsIn : DynamicsPort;
    port engineOut : EngineDemandPort;
  }

  part def FuelConsumptionAnalysis {
    attribute fuelConsumedLiters : Real;
    attribute fuelEconomyMPG : Real;
    attribute fuelEconomyLper100km : Real;
    attribute requiredMPG : Real = 30;
    attribute meetsRequirement : Boolean;
    port engineIn : EngineDemandPort;
    port fuelOut : FuelEconomyPort;
  }

  part def FuelEconomySimulationSystem {
    part vehicle : VehicleParameters;
    part scenario : DrivingScenario;
    part dynamics : DynamicsAnalysis;
    part engine : EngineDemandAnalysis;
    part fuel : FuelConsumptionAnalysis;

    connect vehicle.paramsOut to dynamics.paramsIn;
    connect scenario.scenarioOut to dynamics.scenarioIn;
    connect dynamics.dynamicsOut to engine.dynamicsIn;
    connect engine.engineOut to fuel.engineIn;
  }

  part system : FuelEconomySimulationSystem;
}