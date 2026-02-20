package VehicleDynamicsModel {
    private import ScalarValues::*;
    
    part def Vehicle {
        attribute mass : Real;
        attribute wheelPower : Real;
        attribute dragCoefficient : Real;
        attribute rollingResistanceCoefficient : Real;
        attribute initialSpeed : Real;
        attribute initialPosition : Real;
        attribute currentSpeed : Real;
        attribute currentPosition : Real;
        attribute currentAcceleration : Real;
    }
}