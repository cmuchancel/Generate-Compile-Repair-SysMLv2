package VehicleModel {
    private import ScalarValues::*;

    enum def Gear {
        enum neutral;
        enum second;
    }

    part def Driver {
        attribute name : String;
    }

    part def Engine {
    }

    part def Vehicle {
        attribute mass : Real;
        attribute gear : Gear;
        attribute throttle : Real;
        part engine : Engine;
        part driver : Driver;
    }

    part Alice : Driver {
        attribute name : String = "Alice";
    }

    part Bob : Driver {
        attribute name : String = "Bob";
    }

    part engine1 : Engine;

    part vehicleAtTime1 : Vehicle {
        attribute mass : Real = 2000;
        attribute gear : Gear = Gear::neutral;
        attribute throttle : Real = 0.0;
        part engine : Engine = engine1;
        part driver : Driver = Alice;
    }

    part vehicleAtTime2 : Vehicle {
        attribute mass : Real = 1500;
        attribute gear : Gear = Gear::second;
        attribute throttle : Real = 0.5;
        part engine : Engine = engine1;
        part driver : Driver = Bob;
    }
}