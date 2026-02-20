package VehicleModel {
    private import ScalarValues::*;

    enum def GearPosition {
        enum NEUTRAL;
        enum SECOND;
    }

    part def Engine;

    part def Vehicle {
        attribute mass : Real;
        attribute gear : GearPosition;
        attribute throttle : Real;
        part engine : Engine;
    }

    part def Driver {
        attribute name : String;
    }

    part def TimeSlice {
        attribute mass : Real;
        attribute gear : GearPosition;
        attribute throttle : Real;
        part assignedDriver : Driver;
    }

    part alice : Driver {
        attribute name : String = "Alice";
    }

    part bob : Driver {
        attribute name : String = "Bob";
    }

    part slice1 : TimeSlice {
        attribute mass : Real = 2000.0;
        attribute gear : GearPosition = GearPosition::NEUTRAL;
        attribute throttle : Real = 0.0;
        part assignedDriver : Driver;
    }

    part slice2 : TimeSlice {
        attribute mass : Real = 1500.0;
        attribute gear : GearPosition = GearPosition::SECOND;
        attribute throttle : Real = 0.5;
        part assignedDriver : Driver;
    }

    part myVehicle : Vehicle {
        attribute mass : Real = 2000.0;
        attribute gear : GearPosition = GearPosition::NEUTRAL;
        attribute throttle : Real = 0.0;
        part engine : Engine;
    }
}