package VehicleSystem {
    private import ScalarValues::*;

    enum def VehicleState {
        enum NEUTRAL;
        enum DRIVE;
        enum REVERSE;
        enum PARK;
    }

    part def Engine {
        attribute power : Real;
    }

    part def Driver {
        attribute name : String;
    }

    part def Vehicle {
        attribute mass : Real;
        attribute gear : Integer;
        attribute acceleratorPosition : Real;
        attribute state : VehicleState = VehicleState::NEUTRAL;

        part engine : Engine;
        part driver : Driver;
    }

    part Vehicle_1 : Vehicle {
        attribute mass : Real = 2000.0;
        attribute gear : Integer = 0;
        attribute acceleratorPosition : Real = 0.0;
    }

    part Vehicle_1_next : Vehicle {
        attribute mass : Real = 1500.0;
        attribute gear : Integer = 2;
        attribute acceleratorPosition : Real = 0.5;
    }
}