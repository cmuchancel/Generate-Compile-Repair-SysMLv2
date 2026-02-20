package VehicleDynamicsModel {
  private import ScalarValues::*;

  port def DynamicsParameters {
    attribute wheelPower : Real;
    attribute dragCoefficient : Real;
    attribute frictionCoefficient : Real;
    attribute mass : Real;
    attribute speed : Real;
    attribute time : Real;
    attribute initialSpeed : Real;
    attribute availablePower : Real;
    attribute effectivePower : Real;
    attribute acceleration : Real;
    attribute calculatedSpeed : Real;
    attribute displacement : Real;
    attribute position : Real;
  }

  part def PowerCalculator {
    port io : DynamicsParameters;
  }

  part def AccelerationCalculator {
    port io : DynamicsParameters;
  }

  part def SpeedCalculator {
    port io : DynamicsParameters;
  }

  part def DisplacementCalculator {
    port io : DynamicsParameters;
  }

  part def VehicleDynamicsSystem {
    port input : DynamicsParameters;
    port output : DynamicsParameters;
    part powerCalc : PowerCalculator;
    part accelCalc : AccelerationCalculator;
    part speedCalc : SpeedCalculator;
    part dispCalc : DisplacementCalculator;
    connect input to powerCalc.io;
    connect powerCalc.io to accelCalc.io;
    connect accelCalc.io to speedCalc.io;
    connect speedCalc.io to dispCalc.io;
    connect dispCalc.io to output;
  }

  part vehicleDynamics : VehicleDynamicsSystem;
}