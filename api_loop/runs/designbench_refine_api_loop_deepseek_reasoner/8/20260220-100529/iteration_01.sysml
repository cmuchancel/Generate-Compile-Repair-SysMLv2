package VehicleModel {
    private import ScalarValues::*;

    enum def FuelType {
        enum gasoline;
        enum diesel;
        enum electric;
    }

    item def Fuel {
        attribute type : FuelType;
    }

    part def Person {
        attribute name : String;
        attribute licenseNumber : String;
    }

    part def FuelTank {
        attribute capacity : Real;
        attribute currentLevel : Real;
        part storedFuel : Fuel;
    }

    part def Vehicle {
        attribute model : String;
        attribute weight : Real;
        attribute color : String;
        part driver : Person;
        part tank : FuelTank;
    }
}