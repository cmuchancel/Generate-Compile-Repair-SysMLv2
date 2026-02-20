package VehicleSystem {
    private import ScalarValues::*;

    enum def GearSetting {
        enum PARK;
        enum REVERSE;
        enum NEUTRAL;
        enum DRIVE;
    }

    part def Engine {
        attribute serialNumber : String;
        attribute horsepower : Real;
        attribute isRunning : Boolean = false;
    }

    part def Driver {
        attribute name : String;
        attribute licenseNumber : String;
        attribute experienceYears : Integer = 0;
    }

    part def VehicleStatus {
        attribute currentGear : GearSetting = GearSetting::PARK;
        attribute acceleratorPosition : Real = 0.0;
    }

    part def Vehicle {
        attribute mass : Real;

        part status : VehicleStatus;
        part engine : Engine;
        part driver : Driver [0..1];
    }
}