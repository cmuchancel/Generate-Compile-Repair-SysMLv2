package VehicleLongitudinalDynamics {
  private import ScalarValues::*;

  part def VehicleSystem {
    attribute initialPosition : Real;
    attribute initialVelocity : Real;
    attribute mass : Real;
    attribute drivePower : Real;
    attribute timeInterval : Real;
    attribute dragCoefficient : Real;
    attribute frictionCoefficient : Real;
    attribute acceleration : Real;
    attribute finalVelocity : Real;
    attribute finalPosition : Real;
  }

  part VehicleSim : VehicleSystem;
}