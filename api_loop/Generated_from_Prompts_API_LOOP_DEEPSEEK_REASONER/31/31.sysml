package VehicleSimulation {
    private import ScalarValues::*;

    item def PowerSequence {
        attribute values : Real[*];
    }

    part def Vehicle {
        attribute mass : Real;
        attribute position : Real;
        attribute velocity : Real;
        attribute timeStep : Real;
    }

    part def Simulation {
        attribute vehicle : Vehicle;
        attribute powerSequence : PowerSequence;
        attribute trajectory : Real[*];
    }
}