package VehicleSystem {
    private import ScalarValues::*;

    part def VehicleState {
        attribute gear : Integer;
        attribute accelerator : Real;
    }

    part def Vehicle {
        attribute mass : Real;
        part vehicleState : VehicleState;
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

    part Vehicle_1_initial : Vehicle {
        attribute mass = 2000.0;
        part vehicleState : VehicleState {
            attribute gear = 0;
            attribute accelerator = 0.0;
        }
    }

    part Vehicle_1_next : Vehicle {
        attribute mass = 1500.0;
        part vehicleState : VehicleState {
            attribute gear = 2;
            attribute accelerator = 0.5;
        }
    }
}