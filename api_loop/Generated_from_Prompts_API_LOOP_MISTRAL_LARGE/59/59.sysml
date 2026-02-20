package VehicleDynamicsSimulation {
    private import ScalarValues::*;

    enum def ResistanceType {
        enum Aerodynamic;
        enum Rolling;
    }

    part def Vehicle {
        attribute mass : Real;
        attribute initialSpeed : Real;
        attribute initialPosition : Real;
        attribute dragCoefficient : Real;
        attribute rollingResistanceCoefficient : Real;
        attribute wheelEndPower : Real;
        attribute timeStep : Real;

        attribute currentSpeed : Real = initialSpeed;
        attribute currentPosition : Real = initialPosition;
        attribute currentAcceleration : Real;

        port powerInput : PowerPort;
        port stateOutput : StatePort;
    }

    part def Environment {
        attribute airDensity : Real = 1.225;
        attribute gravitationalAcceleration : Real = 9.81;
    }

    part def SimulationController {
        port powerOutput : PowerPort;
        port stateInput : StatePort;
    }

    port def PowerPort {
        attribute power : Real;
    }

    port def StatePort {
        attribute speed : Real;
        attribute position : Real;
        attribute acceleration : Real;
    }

    item def PowerData {
        attribute value : Real;
    }

    item def StateData {
        attribute speed : Real;
        attribute position : Real;
        attribute acceleration : Real;
    }

    part simulationSystem {
        part vehicle : Vehicle;
        part environment : Environment;
        part controller : SimulationController;

        connect controller.powerOutput to vehicle.powerInput;
        connect vehicle.stateOutput to controller.stateInput;
    }
}