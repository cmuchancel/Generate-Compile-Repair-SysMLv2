package VehicleDynamics {
    private import ScalarValues::*;

    enum def OperatingCondition {
        enum Urban;
        enum Highway;
        enum OffRoad;
        enum Track;
    }

    part def Vehicle {
        attribute mass : Real;
        attribute aerodynamicDragCoefficient : Real;
        attribute frictionCoefficient : Real;
        attribute initialSpeed : Real = 0.0;
        attribute initialPosition : Real = 0.0;
    }

    part def PowerSystem {
        attribute wheelPower : Real;
        attribute actualPowerOutput : Real;
    }

    part def DynamicsCalculator {
        port inputPower : PowerInput;
        port inputParameters : VehicleParameters;
        port outputDynamics : DynamicsOutput;

        attribute currentSpeed : Real;
        attribute currentAcceleration : Real;
        attribute currentPosition : Real;
        attribute timeStep : Real = 0.1;
    }

    port def PowerInput {
        attribute power : Real;
    }

    port def VehicleParameters {
        attribute mass : Real;
        attribute dragCoefficient : Real;
        attribute frictionCoefficient : Real;
        attribute currentSpeed : Real;
    }

    port def DynamicsOutput {
        attribute acceleration : Real;
        attribute speed : Real;
        attribute position : Real;
    }

    part vehicle : Vehicle;
    part powerSystem : PowerSystem;
    part calculator : DynamicsCalculator;

    connect powerSystem to calculator.inputPower;
    connect vehicle to calculator.inputParameters;
    connect calculator.outputDynamics to vehicle;
}