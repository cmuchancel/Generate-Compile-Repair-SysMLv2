package VehicleSystem {
    private import ScalarValues::*;

    enum def OperationalStatus {
        enum ACTIVE;
        enum INACTIVE;
    }

    enum def GearSetting {
        enum PARK;
        enum REVERSE;
        enum NEUTRAL;
        enum DRIVE;
    }

    part def Wheel {
        attribute wheelId : String;
        attribute diameter : Real;
    }

    part def Engine {
        attribute horsepower : Real;
        attribute engineId : String;
    }

    part def Driver {
        attribute driverId : String;
        attribute name : String;
    }

    part def VehicleStatus {
        attribute gearSetting : GearSetting;
        attribute acceleratorPedalPosition : Real;
    }

    part def Vehicle {
        attribute mass : Real;
        attribute status : OperationalStatus;

        part vehicleStatus : VehicleStatus;
        part engine : Engine;
        part driver : Driver;
        part leftFrontWheel : Wheel;
        part rightFrontWheel : Wheel;
    }

    part def WheelSnapshot {
        attribute snapshotTime : String;
        attribute wheelId : String;
        attribute diameter : Real;
        attribute notes : String;
    }

    part def VehicleSnapshot {
        attribute snapshotTime : String;
        attribute mass : Real;
        attribute status : OperationalStatus;
        part leftFrontWheelSnap : WheelSnapshot;
        part rightFrontWheelSnap : WheelSnapshot;
    }

    part vehicle_1 : Vehicle {
        attribute redefines mass = 1500.0;
        attribute redefines status = OperationalStatus::ACTIVE;

        part redefines vehicleStatus : VehicleStatus {
            attribute redefines gearSetting = GearSetting::DRIVE;
            attribute redefines acceleratorPedalPosition = 0.35;
        }

        part redefines engine : Engine {
            attribute redefines horsepower = 200.0;
            attribute redefines engineId = "ENG-001";
        }

        part redefines driver : Driver {
            attribute redefines driverId = "DRV-001";
            attribute redefines name = "John Doe";
        }

        part redefines leftFrontWheel : Wheel {
            attribute redefines wheelId = "WHL-LF-001";
            attribute redefines diameter = 18.0;
        }

        part redefines rightFrontWheel : Wheel {
            attribute redefines wheelId = "WHL-RF-001";
            attribute redefines diameter = 18.0;
        }
    }

    part snapshot_t1 : VehicleSnapshot {
        attribute redefines snapshotTime = "T1";
        attribute redefines mass = 1500.0;
        attribute redefines status = OperationalStatus::ACTIVE;

        part redefines leftFrontWheelSnap : WheelSnapshot {
            attribute redefines snapshotTime = "T1";
            attribute redefines wheelId = "WHL-LF-001";
            attribute redefines diameter = 18.0;
            attribute redefines notes = "Normal condition at T1";
        }

        part redefines rightFrontWheelSnap : WheelSnapshot {
            attribute redefines snapshotTime = "T1";
            attribute redefines wheelId = "WHL-RF-001";
            attribute redefines diameter = 18.0;
            attribute redefines notes = "Normal condition at T1";
        }
    }

    part snapshot_t2 : VehicleSnapshot {
        attribute redefines snapshotTime = "T2";
        attribute redefines mass = 1480.0;
        attribute redefines status = OperationalStatus::ACTIVE;

        part redefines leftFrontWheelSnap : WheelSnapshot {
            attribute redefines snapshotTime = "T2";
            attribute redefines wheelId = "WHL-LF-001";
            attribute redefines diameter = 17.8;
            attribute redefines notes = "Slight wear at T2";
        }

        part redefines rightFrontWheelSnap : WheelSnapshot {
            attribute redefines snapshotTime = "T2";
            attribute redefines wheelId = "WHL-RF-001";
            attribute redefines diameter = 17.9;
            attribute redefines notes = "Minimal wear at T2";
        }
    }
}