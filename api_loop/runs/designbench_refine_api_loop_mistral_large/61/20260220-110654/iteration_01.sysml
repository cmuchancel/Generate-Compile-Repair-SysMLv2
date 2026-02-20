package VehicleLongitudinalDynamics {

    private import ScalarValues::*;

    enum def ResistanceType {
        enum AirResistance;
        enum FrictionResistance;
    }

    part def Vehicle {
        attribute initialPosition : Real;
        attribute initialVelocity : Real;
        attribute mass : Real;
        attribute wheelEndDrivePower : Real;
        attribute dragCoefficient : Real;
        attribute frictionCoefficient : Real;
        attribute timeInterval : Real;
        attribute acceleration : Real;
        attribute finalVelocity : Real;
        attribute finalPosition : Real;
    }

    part def Environment {
        attribute airDensity : Real = 1.225;
        attribute gravitationalAcceleration : Real = 9.81;
    }

    part def DynamicsCalculator {
        port input : InputPort;
        port output : OutputPort;
    }

    port def InputPort {
        attribute position : Real;
        attribute velocity : Real;
        attribute power : Real;
        attribute mass : Real;
        attribute dragCoeff : Real;
        attribute frictionCoeff : Real;
        attribute time : Real;
    }

    port def OutputPort {
        attribute acceleration : Real;
        attribute finalVelocity : Real;
        attribute finalPosition : Real;
    }

    item def ResistanceForce {
        attribute type : ResistanceType;
        attribute value : Real;
    }

    part system {
        part vehicle : Vehicle;
        part environment : Environment;
        part calculator : DynamicsCalculator;

        connect vehicle to calculator.input;
        connect calculator.output to vehicle;
    }
}