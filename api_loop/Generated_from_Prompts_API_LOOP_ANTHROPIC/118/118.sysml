package VehicleManagementSystem {
    private import ScalarValues::*;

    enum def MassGrade {
        enum GRADE_2000;
        enum GRADE_2500;
    }

    enum def FuelStatus {
        enum FULL;
        enum EMPTY;
    }

    enum def ReliabilityStatus {
        enum TRACKED;
        enum FULFILLED;
    }

    part def Engine {
        attribute torqueOutput : Real;
        attribute isConnected : Boolean;
        port drivePort : DriveInterface;
    }

    part def Transmission {
        attribute gearRatio : Real;
        port drivePort : DriveInterface;
    }

    port def DriveInterface {
        attribute transferRate : Real;
    }

    part def Vehicle {
        attribute actualMass : Real;
        attribute maxMass : Real;
        attribute massGrade : MassGrade;
        attribute fuelStatus : FuelStatus;
        attribute reliability : ReliabilityStatus;
        attribute isMassCompliant : Boolean;
        attribute isPowerDeliveryEnabled : Boolean;

        part engine : Engine;
        part transmission : Transmission;

        connect engine.drivePort to transmission.drivePort;
    }

    part vehicleA : Vehicle {
        attribute redefines actualMass = 2000.0;
        attribute redefines maxMass = 2000.0;
        attribute redefines massGrade = MassGrade::GRADE_2000;
        attribute redefines fuelStatus = FuelStatus::FULL;
        attribute redefines isMassCompliant = true;
        attribute redefines isPowerDeliveryEnabled = true;
        attribute redefines reliability = ReliabilityStatus::TRACKED;

        part redefines engine : Engine {
            attribute redefines torqueOutput = 300.0;
            attribute redefines isConnected = true;
        }

        part redefines transmission : Transmission {
            attribute redefines gearRatio = 3.5;
        }
    }

    part vehicleB : Vehicle {
        attribute redefines actualMass = 2500.0;
        attribute redefines maxMass = 2500.0;
        attribute redefines massGrade = MassGrade::GRADE_2500;
        attribute redefines fuelStatus = FuelStatus::EMPTY;
        attribute redefines isMassCompliant = true;
        attribute redefines isPowerDeliveryEnabled = true;
        attribute redefines reliability = ReliabilityStatus::FULFILLED;

        part redefines engine : Engine {
            attribute redefines torqueOutput = 450.0;
            attribute redefines isConnected = true;
        }

        part redefines transmission : Transmission {
            attribute redefines gearRatio = 4.0;
        }
    }
}