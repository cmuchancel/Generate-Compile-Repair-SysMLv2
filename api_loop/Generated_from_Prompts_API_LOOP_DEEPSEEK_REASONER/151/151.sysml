package TransportationSystem {
    private import ScalarValues::*;

    enum def UserRole {
        enum DRIVER;
        enum PASSENGER;
    }

    part def Vehicle {
        attribute maxPassengers : Integer = 4;
        attribute currentPassengerCount : Integer = 0;
        attribute fuelLevel : Real;
        attribute maxTankCapacity : Real;
        port vehiclePort : VehiclePort;
    }

    port def VehiclePort;

    part def Person {
        attribute role : UserRole;
        port personPort : PersonPort;
    }

    port def PersonPort;

    part def GasStation {
        port gasStationPort : GasStationPort;
    }

    port def GasStationPort;

    part def Environment;

    part vehicle : Vehicle;
    part driver : Person {
        attribute role : UserRole = UserRole::DRIVER;
    }
    part passengers : Person[0..4] {
        attribute role : UserRole = UserRole::PASSENGER;
    }
    part gasStation : GasStation;
    part environment : Environment;

    connect driver.personPort to vehicle.vehiclePort;
    connect passengers.personPort to vehicle.vehiclePort;
    connect vehicle.vehiclePort to gasStation.gasStationPort;
}