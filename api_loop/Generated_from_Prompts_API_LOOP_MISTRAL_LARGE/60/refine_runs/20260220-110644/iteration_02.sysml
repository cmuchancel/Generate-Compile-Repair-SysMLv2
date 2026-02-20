package VehicleDynamics {
    private import ScalarValues::*;

    enum def ResistanceType {
        enum AERODYNAMIC;
        enum ROLLING;
    }

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
        attribute timeStep : Real = 0.1;
    }

    part def Environment {
        attribute airDensity : Real = 1.225;
        attribute gravitationalAcceleration : Real = 9.81;
    }

    port def ComputePort {
    }

    part def DynamicsCalculator {
        port compute : ComputePort;
    }

    part system {
        part vehicle : Vehicle;
        part environment : Environment;
        part calculator : DynamicsCalculator;

        connect vehicle to calculator.compute;
        connect environment to calculator.compute;
    }
}