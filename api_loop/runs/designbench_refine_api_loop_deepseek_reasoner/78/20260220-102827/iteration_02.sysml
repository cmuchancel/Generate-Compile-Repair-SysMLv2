package TransportationSystem {
    private import ScalarValues::*;

    item def Fuel {
        attribute amount : Real;
        attribute capacity : Real;
    }

    enum def WeatherCondition {
        Sunny;
        Rainy;
        Snowy;
        Foggy;
    }

    enum def RoadCondition {
        Dry;
        Wet;
        Icy;
        Rough;
    }

    part def Environment {
        attribute weather : WeatherCondition;
        attribute roadCondition : RoadCondition;
    }

    part def Person {
        attribute name : String;
    }

    part def Driver {
        part person : Person;
        attribute licenseId : String;
    }

    part def Passenger {
        part person : Person;
    }

    port def EntryExitPort;

    port def RefuelPort;

    part def GasStation {
        port refuel : RefuelPort;
        attribute fuelPrice : Real;
    }

    part def Vehicle {
        attribute fuel : Fuel;
        part driver : Driver [0..1];
        part passengers : Passenger [0..4];
        port entryExit : EntryExitPort;
        port refuel : RefuelPort;
        attribute location : String;
        attribute speed : Real;
        attribute maxSpeed : Real;
    }

    part TransportationScenario {
        part vehicle : Vehicle;
        part environment : Environment;
        part gasStation : GasStation;
        part driver : Driver;
        part passenger1 : Passenger;
        part passenger2 : Passenger;
        part passenger3 : Passenger;
        part passenger4 : Passenger;
        connect driver to vehicle.entryExit;
        connect passenger1 to vehicle.entryExit;
        connect passenger2 to vehicle.entryExit;
        connect passenger3 to vehicle.entryExit;
        connect passenger4 to vehicle.entryExit;
        connect vehicle.refuel to gasStation.refuel;
    }
}