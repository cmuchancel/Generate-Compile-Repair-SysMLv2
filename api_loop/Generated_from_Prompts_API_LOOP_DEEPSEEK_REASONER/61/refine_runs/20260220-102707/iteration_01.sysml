package VehicleDynamics {
    private import ScalarValues::*;

    part def Vehicle {
        attribute initialPosition : Real;
        attribute initialVelocity : Real;
        attribute mass : Real;
        attribute power : Real;
        attribute timeInterval : Real;
        attribute dragCoefficient : Real;
        attribute frictionCoefficient : Real;
        
        attribute acceleration : Real;
        attribute finalVelocity : Real;
        attribute finalPosition : Real;
    }

    part vehicle : Vehicle;
}