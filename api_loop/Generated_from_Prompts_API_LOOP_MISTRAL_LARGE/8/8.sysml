package VehicleModeling {
    private import ScalarValues::*;

    enum def FuelType {
        enum GASOLINE;
        enum DIESEL;
        enum ELECTRIC;
        enum HYDROGEN;
    }

    part def Fuel {
        attribute fuelType : FuelType;
        attribute quantity : Real;
    }

    part def FuelTank {
        attribute capacity : Real;
        part fuel : Fuel;
    }

    part def Driver {
        attribute name : String;
        attribute licenseNumber : String;
        attribute age : Integer;
    }

    part def Vehicle {
        attribute make : String;
        attribute model : String;
        attribute year : Integer;
        attribute weight : Real;

        part driver : Driver;
        part fuelTank : FuelTank;
    }

    part systemContext {
        part vehicle : Vehicle;
        part fuelSupply : Fuel;
    }

    connect systemContext.vehicle.fuelTank.fuel to systemContext.fuelSupply;
}