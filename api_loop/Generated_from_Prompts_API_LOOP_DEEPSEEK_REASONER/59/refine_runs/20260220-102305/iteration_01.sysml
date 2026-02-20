package VehicleDynamics {
    private import ScalarValues::*;

    part def Vehicle {
        attribute mass : Real;
        attribute dragCoefficient : Real;
        attribute rollingResistanceCoefficient : Real;
        attribute frontalArea : Real;
        attribute wheelPower : Real;
        attribute currentSpeed : Real;
        attribute currentPosition : Real;
        attribute currentAcceleration : Real;
    }

    part def StraightLineSimulation {
        attribute timeStep : Real;
        attribute airDensity : Real = 1.225;
        attribute gravity : Real = 9.81;
        part vehicle : Vehicle;
    }
}