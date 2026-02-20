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
        attribute state : VehicleState = VehicleState::on;
    }

    part def Road {
        attribute slopeAngle : Real;
        attribute frictionCoefficient : Real;
    }

    part def SimulationSnapshot {
        attribute time : Real;
        attribute vehicleState : VehicleState;
        attribute vehiclePosition : Real[3];
        attribute vehicleVelocity : Real[3];
        attribute vehicleAcceleration : Real[3];
    }

    part def Simulation {
        part vehicle : Vehicle;
        part road : Road;
        part snapshots : SimulationSnapshot[*];
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