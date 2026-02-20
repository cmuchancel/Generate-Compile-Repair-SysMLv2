package VehicleModel {
    private import ScalarValues::*;

    enum def GearSetting {
        enum Park;
        enum Reverse;
        enum Neutral;
        enum Drive;
    }

    part def OperationalStatus {
        attribute gearSetting : GearSetting;
        attribute acceleratorPedalPosition : Real;
    }

    part def Engine {
    }

    part def Driver {
    }

    part def Wheel {
    }

    part def Vehicle {
        attribute mass : Real;
        attribute isOperational : Boolean;
        part status : OperationalStatus;
        part engine : Engine;
        ref part driver : Driver;
        part leftFrontWheel : Wheel;
        part rightFrontWheel : Wheel;
        part leftRearWheel : Wheel;
        part rightRearWheel : Wheel;
    }

    part vehicle_1 : Vehicle {
        part leftFrontWheel_snapshot1 : Wheel;
        part rightFrontWheel_snapshot1 : Wheel;
        part leftFrontWheel_snapshot2 : Wheel;
        part rightFrontWheel_snapshot2 : Wheel;
    }
}