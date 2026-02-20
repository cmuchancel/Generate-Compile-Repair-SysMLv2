package VehicleSystem {
    private import ScalarValues::*;

    enum def OperationalStatus {
        enum NEUTRAL
        enum DRIVE
        enum REVERSE
        enum PARK
    }

    part def Engine {
        attribute mass : Real;
    }

    part def Wheel {
        attribute mass : Real;
    }

    part def Driver {
        attribute name : String;
    }

    part def Vehicle {
        attribute mass : Real;
        attribute operationalStatus : OperationalStatus = OperationalStatus::NEUTRAL;
        attribute gear : OperationalStatus = OperationalStatus::NEUTRAL;
        attribute acceleratorPedalPosition : Real = 0.0;

        part engine : Engine;
        part driver : Driver;
        part leftFrontWheel : Wheel;
        part rightFrontWheel : Wheel;
    }

    part vehicle_1 : Vehicle {
        attribute mass : Real = 1500.0;
        part leftFrontWheel : Wheel {
            attribute mass : Real = 20.0;
        }
        part rightFrontWheel : Wheel {
            attribute mass : Real = 20.0;
        }
    }

    // Time snapshot 1
    part vehicle_1_snapshot1 {
        attribute time : Real = 0.0;
        part vehicleRef : Vehicle = vehicle_1;
        part leftFrontWheelRef : Wheel = vehicle_1.leftFrontWheel;
        part rightFrontWheelRef : Wheel = vehicle_1.rightFrontWheel;
    }

    // Time snapshot 2
    part vehicle_1_snapshot2 {
        attribute time : Real = 1.0;
        part vehicleRef : Vehicle = vehicle_1;
        part leftFrontWheelRef : Wheel = vehicle_1.leftFrontWheel;
        part rightFrontWheelRef : Wheel = vehicle_1.rightFrontWheel;
    }
}