package VehicleSpatiotemporalSimulation {
    private import ScalarValues::*;

    enum def VehicleState {
        enum on;
        enum driving;
        enum off;
    }

    part def Vehicle {
        attribute mass : Real;
        attribute position : Real[3];
        attribute velocity : Real[3];
        attribute acceleration : Real[3];
        attribute vehicleState : VehicleState = VehicleState::on;
    }

    part def Road {
        attribute slopeAngle : Real;
        attribute frictionCoefficient : Real;
    }

    part def SimulationSnapshot {
        attribute snapshotTime : Real;
        attribute snapshotState : VehicleState;
        attribute snapshotPosition : Real[3];
        attribute snapshotVelocity : Real[3];
        attribute snapshotAcceleration : Real[3];
    }

    part def Simulation {
        part vehicle : Vehicle;
        part road : Road;
        part snapshotSeries : SimulationSnapshot[*];
    }

    item def TimePoint {
        attribute value : Real;
    }

    port def VehiclePort {
        attribute currentState : VehicleState;
    }

    part simulation : Simulation;

    connect simulation.vehicle to simulation.road;
}