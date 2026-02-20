package TransportationSystem {
    private import ScalarValues::*;

    enum def FuelStatus {
        enum FULL;
        enum ADEQUATE;
        enum LOW;
        enum EMPTY;
    }

    enum def OccupantRole {
        enum DRIVER;
        enum PASSENGER;
    }

    enum def TravelPhase {
        enum BOARDING;
        enum TRAVELING;
        enum REFUELING;
        enum ALIGHTING;
        enum COMPLETED;
    }

    item def Fuel {
        attribute amount : Real;
    }

    item def Person {
        attribute role : OccupantRole;
        attribute name : String;
    }

    part def GasStation {
        attribute stationName : String;
        attribute fuelAvailable : Real;
        port fuelDispensePort : FuelPort;
    }

    part def FuelTank {
        attribute maxCapacity : Real;
        attribute currentLevel : Real;
        attribute lowFuelThreshold : Real;
        attribute fuelStatus : FuelStatus;
        port tankFillPort : FuelPort;
    }

    part def Engine {
        attribute isRunning : Boolean;
        attribute fuelConsumptionRate : Real;
    }

    part def PassengerCompartment {
        attribute maxPassengers : Integer;
        attribute currentPassengerCount : Integer;
        port boardingPort : OccupantPort;
        port alightingPort : OccupantPort;
    }

    part def DriverSeat {
        attribute isOccupied : Boolean;
        port driverBoardPort : OccupantPort;
        port driverAlightPort : OccupantPort;
    }

    part def EnvironmentSensor {
        attribute temperature : Real;
        attribute weatherCondition : String;
        attribute roadCondition : String;
    }

    part def NavigationSystem {
        attribute currentLocation : String;
        attribute destination : String;
        attribute routeDistance : Real;
    }

    port def FuelPort {
        attribute flowRate : Real;
    }

    port def OccupantPort {
        attribute capacity : Integer;
    }

    part def Vehicle {
        attribute vehicleId : String;
        attribute maxOccupants : Integer = 5;
        attribute currentPhase : TravelPhase;
        attribute isOperational : Boolean;

        part fuelTank : FuelTank;
        part engine : Engine;
        part compartment : PassengerCompartment;
        part driverSeat : DriverSeat;
        part envSensor : EnvironmentSensor;
        part navSystem : NavigationSystem;

        port vehicleFuelPort : FuelPort;
        port passengerBoardPort : OccupantPort;
        port passengerAlightPort : OccupantPort;
        port driverBoardPortVehicle : OccupantPort;
        port driverAlightPortVehicle : OccupantPort;

        connect vehicleFuelPort to fuelTank.tankFillPort;
        connect passengerBoardPort to compartment.boardingPort;
        connect passengerAlightPort to compartment.alightingPort;
        connect driverBoardPortVehicle to driverSeat.driverBoardPort;
        connect driverAlightPortVehicle to driverSeat.driverAlightPort;
    }

    part def TransportationService {
        attribute serviceId : String;
        attribute totalDistance : Real;
        attribute fuelLowThresholdPercent : Real = 10.0;
        attribute maxPassengerCount : Integer = 4;
        attribute driverCount : Integer = 1;
        attribute refuelingRequired : Boolean;
        attribute tripCompleted : Boolean;

        part vehicle : Vehicle;
        part station : GasStation;

        connect vehicle.vehicleFuelPort to station.fuelDispensePort;
    }
}