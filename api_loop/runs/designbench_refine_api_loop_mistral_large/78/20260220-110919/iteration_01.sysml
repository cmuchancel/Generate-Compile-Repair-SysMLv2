package TransportationSystem {
    private import ScalarValues::*;

    enum def VehicleState {
        parked
        driving
        refueling
        arrived
    }

    enum def EnvironmentCondition {
        clear
        rainy
        snowy
        foggy
    }

    part def Person {
        attribute role : String;
    }

    part def Driver specializes Person {
        attribute licenseValid : Boolean = true;
    }

    part def Passenger specializes Person {
    }

    part def Vehicle {
        attribute state : VehicleState = VehicleState::parked;
        attribute fuelLevel : Real = 100.0;
        attribute maxPassengers : Integer = 4;
        attribute currentPassengers : Integer = 0;

        part driver : Driver;
        part passengers : Passenger[0..4];

        port def FuelPort;
        port fuelIn : FuelPort;

        port def ControlPort;
        port control : ControlPort;
    }

    part def GasStation {
        port def FuelDispenserPort;
        port dispenser : FuelDispenserPort;
    }

    part def Environment {
        attribute condition : EnvironmentCondition = EnvironmentCondition::clear;
    }

    part def TransportationScenario {
        part vehicle : Vehicle;
        part gasStation : GasStation;
        part environment : Environment;
        part driver : Driver;
        part passengers : Passenger[0..4];

        connect vehicle.driver to driver;
        connect vehicle.passengers to passengers;
        connect vehicle.fuelIn to gasStation.dispenser;
    }

    item def Fuel;
}