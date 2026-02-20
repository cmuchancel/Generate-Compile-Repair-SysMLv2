package VehicleSystem {
    private import ScalarValues::*;

    enum def GearSetting {
        enum PARK;
        enum REVERSE;
        enum NEUTRAL;
        enum DRIVE;
    }

    part def OperationalStatus {
        attribute gear : GearSetting;
        attribute acceleratorPedalPosition : Real;
    }

    part def Engine {
        attribute isRunning : Boolean;
    }

    part def Wheel {
        attribute position : String;
        attribute rotationSpeed : Real;
    }

    port def DriverInterface;

    part def Driver {
        port vehicleInterface : DriverInterface;
    }

    part def Vehicle {
        attribute mass : Real;
        attribute name : String;

        part status : OperationalStatus;
        part powerplant : Engine;
        port driverPort : DriverInterface;

        part leftFrontWheel : Wheel;
        part rightFrontWheel : Wheel;
    }

    part vehicle_1 : Vehicle {
        attribute mass : Real = 1500.0;
        attribute name : String = "vehicle_1";

        part status : OperationalStatus {
            attribute gear : GearSetting = GearSetting::PARK;
            attribute acceleratorPedalPosition : Real = 0.0;
        }

        part powerplant : Engine {
            attribute isRunning : Boolean = false;
        }

        part leftFrontWheel : Wheel {
            attribute position : String = "left front";
            attribute rotationSpeed : Real = 0.0;
        }

        part rightFrontWheel : Wheel {
            attribute position : String = "right front";
            attribute rotationSpeed : Real = 0.0;
        }

        port driverPort : DriverInterface;
    }

    part driver_1 : Driver {
        port vehicleInterface : DriverInterface;
    }

    connect vehicle_1.driverPort to driver_1.vehicleInterface;
}