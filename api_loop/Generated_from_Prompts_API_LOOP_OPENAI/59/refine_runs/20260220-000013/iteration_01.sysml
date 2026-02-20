package VehicleDynamicsPackage {
  private import ScalarValues::*;

  part def VehicleDynamicCalculator {
    attribute wheelEndPower : Real;
    attribute totalMass : Real;
    attribute dragCoefficient : Real;
    attribute rollingResistanceCoefficient : Real;
    attribute frontalArea : Real;
    attribute airDensity : Real;
    attribute timeStep : Real;
    attribute initialSpeed : Real;
    attribute initialPosition : Real;
    attribute availableDrivingPower : Real;
    attribute acceleration : Real;
    attribute newSpeed : Real;
    attribute newPosition : Real;
  }

  part def VehicleDynamicsSystem {
    part calculator : VehicleDynamicCalculator;
  }
}