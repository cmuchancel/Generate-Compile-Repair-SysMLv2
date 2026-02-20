package VehicleLongitudinalDynamics {
    private import ScalarValues::*;

    enum def OperatingCondition {
        enum accelerating;
        enum decelerating;
        enum cruising;
    }

    item def PowerSegment {
        attribute power : Real;
        attribute duration : Real;
    }

    part def Vehicle {
        attribute mass : Real;
        attribute initialPosition : Real = 0.0;
        attribute initialVelocity : Real = 0.0;
        attribute currentAcceleration : Real;
        attribute currentVelocity : Real;
        attribute currentPosition : Real;
        attribute condition : OperatingCondition;
    }

    part def SimulationEngine {
        attribute timeStep : Real;
        attribute powerSegments : PowerSegment[*];
        attribute accelerationSequence : Real[*];
        attribute velocitySequence : Real[*];
        attribute positionSequence : Real[*];
        part vehicle : Vehicle;
    }

    part def VehicleSimulationSystem {
        part vehicle : Vehicle;
        part engine : SimulationEngine;
        connect engine to vehicle;
    }
}