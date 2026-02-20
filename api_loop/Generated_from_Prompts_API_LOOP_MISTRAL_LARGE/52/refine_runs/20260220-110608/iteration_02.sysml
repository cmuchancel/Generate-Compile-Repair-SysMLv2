package VehicleSystem {
    private import ScalarValues::*;

    enum def GearSetting {
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
        attribute gearState : GearSetting = GearSetting::NEUTRAL;

        part engine : Engine;
        part driver : Driver;
    }

    part Vehicle_1 :> Vehicle {
        mass = 2000.0;
        gear = 0;
        acceleratorPosition = 0.0;
        gearState = GearSetting::NEUTRAL;
    }

    part Vehicle_1_next :> Vehicle {
        mass = 1500.0;
        gear = 2;
        acceleratorPosition = 0.5;
        gearState = GearSetting::DRIVE;
    }
}