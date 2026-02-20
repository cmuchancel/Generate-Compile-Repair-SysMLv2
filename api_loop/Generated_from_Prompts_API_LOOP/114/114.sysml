package VehicleRoadSimulation {
    private import ScalarValues::*;

    enum def VehiclePowerState {
        enum On;
        enum Off;
    }

    part def Vehicle {
        attribute mass : Real;
        attribute position : Real;
        attribute velocity : Real;
        attribute acceleration : Real;
        attribute powerState : VehiclePowerState;
    }

    part def Road {
        attribute slopeAngle : Real;
        attribute frictionCoefficient : Real;
    }

    part def Snapshot {
        attribute time : Real;
        attribute vehicleMass : Real;
        attribute vehiclePosition : Real;
        attribute vehicleVelocity : Real;
        attribute vehicleAcceleration : Real;
        attribute vehiclePowerState : VehiclePowerState;
        attribute roadSlopeAngle : Real;
        attribute roadFrictionCoefficient : Real;
    }

    part def Simulation {
        attribute startTime : Real;
        attribute endTime : Real;
        attribute snapshotCount : Integer;
        part vehicle : Vehicle;
        part road : Road;
        part startSnapshot : Snapshot;
        part midSnapshot : Snapshot;
        part endSnapshot : Snapshot;
    }

    part systemSimulation : Simulation;
}