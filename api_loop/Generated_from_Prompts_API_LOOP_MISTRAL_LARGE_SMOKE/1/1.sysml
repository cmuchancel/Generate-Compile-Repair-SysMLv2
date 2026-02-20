package VehicleSystem {
    private import ScalarValues::*;

    enum def VehicleStatus {
        enum parked;
        enum driving;
        enum idle;
    }

    enum def GearSetting {
        enum park;
        enum reverse;
        enum neutral;
        enum drive;
        enum low;
    }

    part def Engine {
        attribute displacement : Real;
        attribute horsepower : Real;
        attribute isRunning : Boolean;
    }

    part def Person {
        attribute name : String;
        attribute age : Integer;
    }

    part def Vehicle {
        attribute mass : Real;
        attribute status : VehicleStatus = VehicleStatus::parked;
        attribute gear : GearSetting = GearSetting::park;
        attribute acceleratorPedalPosition : Real = 0.0;
        part engine : Engine;
        part driver : Person;
    }
}