package VehicleSystem {
    private import ScalarValues::*;

    enum def GearSetting {
        enum Park;
        enum Reverse;
        enum Neutral;
        enum Drive;
    }

    part def Engine {
        attribute powerOutput : Real;
        attribute isRunning : Boolean = false;
    }

    part def Driver {
        attribute name : String;
        attribute licenseNumber : String;
    }

    part def Vehicle {
        attribute mass : Real;
        attribute currentGear : GearSetting = GearSetting::Park;
        attribute acceleratorPedalPosition : Real = 0.0;
        part engine : Engine;
        part driver : Driver[0..1];
    }
}