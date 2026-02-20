package VehicleSystem {
    private import ScalarValues::*;

    enum def FuelType {
        enum Gasoline;
        enum Diesel;
        enum Electric;
    }

    item def Fuel {
        attribute fuelType : FuelType;
        attribute quantity : Real;
    }

    item def Person {
        attribute firstName : String;
        attribute lastName : String;
        attribute age : Integer;
    }

    part def FuelTank {
        attribute capacity : Real;
        attribute storedFuelType : FuelType;
        part containedFuel : Fuel;
    }

    part def Driver {
        attribute driverInfo : Person;
        attribute licenseNumber : String;
    }

    part def Vehicle {
        attribute make : String;
        attribute model : String;
        attribute year : Integer;
        attribute mass : Real;
        attribute maxSpeed : Real;
        part vehicleDriver : Driver;
        part vehicleFuelTank : FuelTank;
    }

    part systemRoot {
        part car : Vehicle {
            attribute make = "Toyota";
            attribute model = "Camry";
            attribute year = 2023;
            attribute mass = 1500.0;
            attribute maxSpeed = 200.0;
            part vehicleDriver : Driver {
                attribute licenseNumber = "DL-12345";
                attribute driverInfo : Person {
                    attribute firstName = "John";
                    attribute lastName = "Doe";
                    attribute age = 35;
                }
            }
            part vehicleFuelTank : FuelTank {
                attribute capacity = 60.0;
                attribute storedFuelType = FuelType::Gasoline;
                part containedFuel : Fuel {
                    attribute fuelType = FuelType::Gasoline;
                    attribute quantity = 45.0;
                }
            }
        }
    }
}