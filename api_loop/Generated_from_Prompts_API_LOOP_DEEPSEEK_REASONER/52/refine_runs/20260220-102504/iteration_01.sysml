package VehicleSystem {
    private import ScalarValues::*;

    part def Vehicle {
        attribute mass : Real;
        attribute state : VehicleState;
    }

    part def Engine {
    }

    part def Driver {
    }

    part def CompositeSystem {
        part vehicle : Vehicle;
        part engine : Engine;
        part driver : Driver;
        connect vehicle to engine;
        connect vehicle to driver;
    }

    item def Vehicle_1 : Vehicle {
        attribute mass = 2000.0;
        attribute state = initial;
    }

    item def Vehicle_1_next : Vehicle {
        attribute mass = 1500.0;
        attribute state = adjusted;
    }

    part VehicleState {
        attribute gear : Integer;
        attribute accelerator : Real;
    }

    item def initial : VehicleState {
        attribute gear = 0;
        attribute accelerator = 0.0;
    }

    item def adjusted : VehicleState {
        attribute gear = 2;
        attribute accelerator = 0.5;
    }
}