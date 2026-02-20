package VehicleSystem {
    private import ScalarValues::*;

    enum def GearSetting {
        Park,
        Reverse,
        Neutral,
        Drive
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