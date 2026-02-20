package VehicleRoadSimulation {
    private import ScalarValues::*;

    enum def VehicleState {
        enum on;
        enum driving;
        enum off;
    }

    part def Vehicle {
        attribute mass : Real;
        attribute position : Real;
        attribute velocity : Real;
        attribute acceleration : Real;
        attribute currentState : VehicleState;
    }

    part def Road {
        attribute slope : Real;
        attribute frictionCoefficient : Real;
    }

    item def Snapshot {
        attribute time : Real;
        attribute vehicleState : VehicleState;
        attribute vehiclePosition : Real;
        attribute vehicleVelocity : Real;
        attribute vehicleAcceleration : Real;
        attribute roadSlope : Real;
        attribute roadFriction : Real;
    }

    part def SimulationSystem {
        part mainVehicle : Vehicle;
        part currentRoad : Road;
        attribute recordedSnapshots : Snapshot[*];
    }
}