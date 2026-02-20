package VehicleModel {
    private import ScalarValues::*;
    
    enum def Gear {
        neutral,
        second
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
        name = "Alice";
    }
    
    part Bob : Driver {
        name = "Bob";
    }
    
    part engine1 : Engine;
    
    part vehicleAtTime1 : Vehicle {
        mass = 2000;
        gear = Gear::neutral;
        throttle = 0.0;
        engine = engine1;
        driver = Alice;
    }
    
    part vehicleAtTime2 : Vehicle {
        mass = 1500;
        gear = Gear::second;
        throttle = 0.5;
        engine = engine1;
        driver = Bob;
    }
}