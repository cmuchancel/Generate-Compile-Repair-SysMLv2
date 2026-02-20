package VehicleModel {
    private import ScalarValues::*;

    enum def GearPosition {
        enum NEUTRAL;
        enum FIRST;
        enum SECOND;
        enum THIRD;
        enum FOURTH;
        enum FIFTH;
        enum REVERSE;
    }

    part def Driver {
        attribute name : String;
    }

    part def Engine;

    part def Vehicle {
        attribute mass : Real;
        attribute gear : GearPosition;
        attribute throttle : Real;
        part engine : Engine;
        attribute driver : Driver;
    }

    part alice : Driver {
        attribute name : String = "Alice";
    }

    part bob : Driver {
        attribute name : String = "Bob";
    }

    part vehicle : Vehicle {
        attribute mass : Real = 2000.0;
        attribute gear : GearPosition = GearPosition::NEUTRAL;
        attribute throttle : Real = 0.0;
        part engine : Engine;
        attribute driver : Driver = alice;
    }

    part vehicleSlice1 : Vehicle {
        attribute mass : Real = 2000.0;
        attribute gear : GearPosition = GearPosition::NEUTRAL;
        attribute throttle : Real = 0.0;
        part engine : Engine;
        attribute driver : Driver = alice;
    }

    part vehicleSlice2 : Vehicle {
        attribute mass : Real = 1500.0;
        attribute gear : GearPosition = GearPosition::SECOND;
        attribute throttle : Real = 0.5;
        part engine : Engine;
        attribute driver : Driver = bob;
    }
}