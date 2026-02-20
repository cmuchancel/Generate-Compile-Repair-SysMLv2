package VehicleManagementSystem {
    private import ScalarValues::*;

    enum def GearSetting {
        enum PARK;
        enum REVERSE;
        enum NEUTRAL;
        enum DRIVE;
        enum LOW;
    }

    part def VehicleStatus {
        attribute gearSetting : GearSetting;
        attribute acceleratorPedalPosition : Real;
    }

    part def Engine {
        attribute displacement : Real;
        attribute horsepower : Real;
        attribute rpm : Real;
        attribute isRunning : Boolean;
    }

    part def Person {
        attribute firstName : String;
        attribute lastName : String;
        attribute licenseNumber : String;
    }

    part def Driver {
        attribute employeeId : String;
        attribute experienceYears : Integer;
        part personInfo : Person;
    }

    part def Vehicle {
        attribute vehicleId : String;
        attribute make : String;
        attribute model : String;
        attribute year : Integer;
        attribute mass : Real;
        attribute color : String;
        part engine : Engine;
        part status : VehicleStatus;
        part driver : Driver[0..1];
    }

    part def Fleet {
        part vehicles : Vehicle[*];
        part drivers : Driver[*];
        part engines : Engine[*];
    }

    part fleetInstance : Fleet {
        part vehicles : Vehicle[*] {
            part engine : Engine {
                attribute displacement : Real = 2.0;
                attribute horsepower : Real = 150.0;
                attribute rpm : Real = 3000.0;
                attribute isRunning : Boolean = true;
            }
            part status : VehicleStatus {
                attribute gearSetting : GearSetting = GearSetting::DRIVE;
                attribute acceleratorPedalPosition : Real = 0.3;
            }
            attribute vehicleId : String = "V001";
            attribute make : String = "Toyota";
            attribute model : String = "Camry";
            attribute year : Integer = 2023;
            attribute mass : Real = 1500.0;
            attribute color : String = "Blue";
        }
        part drivers : Driver[*] {
            attribute employeeId : String = "D001";
            attribute experienceYears : Integer = 5;
            part personInfo : Person {
                attribute firstName : String = "John";
                attribute lastName : String = "Doe";
                attribute licenseNumber : String = "LIC123456";
            }
        }
    }
}