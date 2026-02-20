package VehicleSystem {
    private import ScalarValues::*;

    enum def GearState {
        neutral;
        first;
        second;
        third;
        fourth;
        fifth;
        reverse;
    }

    part def Vehicle {
        attribute mass : Real;
        attribute gear : GearState;
        attribute acceleratorPosition : Real;
        part engine : Engine;
        part driver : Driver;
    }

    part def Engine {
        attribute power : Real;
    }

    part def Driver {
        attribute skillLevel : Integer;
    }

    part Vehicle_1 : Vehicle {
        :> mass = 2000;
        :> gear = GearState::neutral;
        :> acceleratorPosition = 0.0;
    }

    part Vehicle_1_updated : Vehicle {
        :> mass = 1500;
        :> gear = GearState::second;
        :> acceleratorPosition = 0.5;
    }
}