package VehicleSystem {
    private import ScalarValues::*;

    enum def GearState {
        enum neutral;
        enum first;
        enum second;
        enum third;
        enum fourth;
        enum fifth;
        enum reverse;
    }

    part def Vehicle {
        attribute mass : Mass;
        attribute gear : GearState;
        attribute acceleratorPosition : Real;
        part engine : Engine;
        part driver : Driver;
    }

    part def Engine {
        attribute power : Power;
    }

    part def Driver {
        attribute skillLevel : Integer;
    }

    part Vehicle_1 : Vehicle {
        mass = 2000;
        gear = GearState::neutral;
        acceleratorPosition = 0.0;
    }

    part Vehicle_1_updated : Vehicle {
        mass = 1500;
        gear = GearState::second;
        acceleratorPosition = 0.5;
    }

    part def VehicleAssembly {
        part vehicle : Vehicle;
        part engine : Engine;
        part driver : Driver;
        
        connect vehicle.engine to engine;
        connect vehicle.driver to driver;
    }
}