package VehicleDynamicsSimulation {
  private import ScalarValues::*;

  part def VehicleParameters {
    attribute mass : Real;
    attribute wheelPower : Real;
    attribute dragCoefficient : Real;
    attribute rollingResistanceCoefficient : Real;
    attribute frontalArea : Real;
  }

  part def InitialConditions {
    attribute initialSpeed : Real;
    attribute initialPosition : Real;
  }

  part def Environment {
    attribute airDensity : Real;
    attribute gravity : Real;
  }

  part def VehicleState {
    attribute time : Real;
    attribute speed : Real;
    attribute position : Real;
    attribute acceleration : Real;
    attribute displacement : Real;
  }

  part def SimulationControl {
    attribute timeStep : Real;
    attribute totalTime : Real;
  }

  part def VehicleDynamicsSystem {
    part parameters : VehicleParameters;
    part initial : InitialConditions;
    part environment : Environment;
    part vehicleState : VehicleState;
    part control : SimulationControl;

    attribute computedDragForce : Real;
    attribute computedRollingResistanceForce : Real;
    attribute computedNetForce : Real;
    attribute computedAcceleration : Real;
  }

  part system : VehicleDynamicsSystem;
}