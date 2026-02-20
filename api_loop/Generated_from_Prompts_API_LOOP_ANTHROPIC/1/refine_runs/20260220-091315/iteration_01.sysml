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
        attribute engineType : String;
    }

    part def Person {
        attribute firstName : String;
        attribute lastName : String;
        attribute licenseNumber : String;
    }

    part def Driver {
        part personInfo : Person;
        attribute experienceYears : Integer;
    }

    part def Vehicle {
        attribute vehicleMass : Real;
        attribute make : String;
        attribute model : String;
        attribute year : Integer;
        part currentStatus : VehicleStatus;
        part engine : Engine;
        part driver : Driver[0..1];
    }

    part def Fleet {
        part vehicles : Vehicle[*];
    }

    part vehicleFleet : Fleet {
        part vehicles : Vehicle[*] {
            part currentStatus : VehicleStatus {
                attribute gearSetting : GearSetting = GearSetting::PARK;
                attribute acceleratorPedalPosition : Real = 0.0;
            }
            part engine : Engine {
                attribute displacement : Real = 2.0;
                attribute horsepower : Real = 150.0;
                attribute engineType : String = "V4";
            }
            part driver : Driver[0..1] {
                part personInfo : Person {
                    attribute firstName : String = "John";
                    attribute lastName : String = "Doe";
                    attribute licenseNumber : String = "DL123456";
                }
                attribute experienceYears : Integer = 5;
            }
            attribute vehicleMass : Real = 1500.0;
            attribute make : String = "Toyota";
            attribute model : String = "Camry";
            attribute year : Integer = 2023;
        }
    }
}