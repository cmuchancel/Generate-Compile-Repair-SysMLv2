package TransportationSystem {
    private import ScalarValues::*;

    enum def RoleType {
        enum DRIVER;
        enum PASSENGER;
    }

    enum def VehicleStatus {
        enum PARKED;
        enum MOVING;
        enum REFUELING;
    }

    enum def JourneyPhase {
        enum BOARDING;
        enum TRAVELING;
        enum ARRIVED;
    }

    item def Person {
        attribute name : String;
        attribute role : RoleType;
    }

    item def Fuel {
        attribute level : Real;
        attribute capacity : Real;
    }

    port def PassengerPort {
        attribute occupantCount : Integer;
    }

    port def FuelPort {
        attribute fuelLevel : Real;
    }

    port def EnvironmentPort {
        attribute condition : String;
    }

    part def Driver {
        attribute name : String;
        attribute role : RoleType = RoleType::DRIVER;
        attribute isOperating : Boolean;
        port driverSeat : PassengerPort;
    }

    part def Passenger {
        attribute name : String;
        attribute role : RoleType = RoleType::PASSENGER;
        attribute isBoarded : Boolean;
        port seat : PassengerPort;
    }

    part def Vehicle {
        attribute status : VehicleStatus;
        attribute maxPassengers : Integer = 4;
        attribute currentSpeed : Real;
        attribute fuelLevel : Real;
        port vehiclePassengerPort : PassengerPort;
        port vehicleFuelPort : FuelPort;
        port vehicleEnvPort : EnvironmentPort;
    }

    part def GasStation {
        attribute location : String;
        attribute availableFuel : Real;
        port stationFuelPort : FuelPort;
    }

    part def Environment {
        attribute weatherCondition : String;
        attribute roadCondition : String;
        port envOutputPort : EnvironmentPort;
    }

    part def StartingPoint {
        attribute locationName : String;
        port boardingPort : PassengerPort;
    }

    part def Destination {
        attribute locationName : String;
        port exitPort : PassengerPort;
    }

    part def TransportationScenario {
        attribute journeyPhase : JourneyPhase;
        attribute journeyComplete : Boolean;

        part theDriver : Driver;

        part passenger1 : Passenger;
        part passenger2 : Passenger;
        part passenger3 : Passenger;
        part passenger4 : Passenger;

        part theVehicle : Vehicle;
        part theGasStation : GasStation;
        part theEnvironment : Environment;
        part theStart : StartingPoint;
        part theDestination : Destination;

        connect theDriver.driverSeat to theVehicle.vehiclePassengerPort;
        connect passenger1.seat to theVehicle.vehiclePassengerPort;
        connect passenger2.seat to theVehicle.vehiclePassengerPort;
        connect passenger3.seat to theVehicle.vehiclePassengerPort;
        connect passenger4.seat to theVehicle.vehiclePassengerPort;
        connect theGasStation.stationFuelPort to theVehicle.vehicleFuelPort;
        connect theEnvironment.envOutputPort to theVehicle.vehicleEnvPort;
        connect theStart.boardingPort to theVehicle.vehiclePassengerPort;
        connect theVehicle.vehiclePassengerPort to theDestination.exitPort;
    }
}