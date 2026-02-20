package TransportationSystem {
    private import ScalarValues::*;

    enum def OccupantRole {
        enum DRIVER;
        enum PASSENGER;
    }

    enum def DoorState {
        enum OPEN;
        enum CLOSED;
    }

    enum def JourneyStatus {
        enum NOT_STARTED;
        enum IN_PROGRESS;
        enum COMPLETED;
    }

    enum def FuelLevel {
        enum FULL;
        enum ADEQUATE;
        enum LOW;
        enum EMPTY;
    }

    enum def SafetyStatus {
        enum SAFE;
        enum UNSAFE;
    }

    enum def EnvironmentCondition {
        enum CLEAR;
        enum RAINY;
        enum FOGGY;
        enum SNOWY;
    }

    part def Person {
        attribute name : String;
        attribute role : OccupantRole;
    }

    part def Driver {
        attribute name : String;
        attribute role : OccupantRole;
        attribute licenseValid : Boolean;
    }

    part def Passenger {
        attribute name : String;
        attribute role : OccupantRole;
        attribute isSeated : Boolean;
    }

    part def Door {
        attribute doorState : DoorState;
        attribute isLocked : Boolean;
    }

    part def Seat {
        attribute isOccupied : Boolean;
        attribute isComfortable : Boolean;
    }

    part def FuelTank {
        attribute capacity : Real;
        attribute currentLevel : FuelLevel;
    }

    part def Engine {
        attribute isRunning : Boolean;
        attribute powerOutput : Real;
    }

    part def SafetySystem {
        attribute status : SafetyStatus;
        attribute seatbeltsFastened : Boolean;
        attribute airbagsFunctional : Boolean;
        attribute brakesOperational : Boolean;
    }

    part def NavigationSystem {
        attribute startPoint : String;
        attribute destination : String;
        attribute routeCalculated : Boolean;
    }

    part def ComfortSystem {
        attribute climateControlActive : Boolean;
        attribute temperature : Real;
        attribute audioSystemActive : Boolean;
    }

    part def EnvironmentSensor {
        attribute condition : EnvironmentCondition;
        attribute visibility : Real;
        attribute roadConditionSafe : Boolean;
    }

    part def GasStation {
        attribute location : String;
        attribute fuelAvailable : Boolean;
    }

    part def Infrastructure {
        attribute roadNetworkAvailable : Boolean;
        attribute gasStationNearby : Boolean;
    }

    port def OccupantPort {
        attribute canEnter : Boolean;
        attribute canExit : Boolean;
    }

    port def FuelPort {
        attribute fuelFlowing : Boolean;
    }

    port def DataPort {
        attribute dataTransmitting : Boolean;
    }

    item def FuelItem {
        attribute amount : Real;
    }

    item def OccupantItem {
        attribute role : OccupantRole;
    }

    item def NavigationData {
        attribute routeInfo : String;
    }

    item def EnvironmentData {
        attribute conditionInfo : String;
    }

    part def Vehicle {
        attribute journeyStatus : JourneyStatus;
        attribute maxPassengers : Integer;
        attribute currentSpeed : Real;
        attribute isAccessible : Boolean;

        part driver : Driver;

        part passengers : Passenger[0..4];

        part frontLeftDoor : Door;
        part frontRightDoor : Door;
        part rearLeftDoor : Door;
        part rearRightDoor : Door;

        part driverSeat : Seat;
        part passengerSeat1 : Seat;
        part passengerSeat2 : Seat;
        part passengerSeat3 : Seat;
        part passengerSeat4 : Seat;

        part engine : Engine;
        part fuelTank : FuelTank;
        part safetySystem : SafetySystem;
        part navigationSystem : NavigationSystem;
        part comfortSystem : ComfortSystem;
        part environmentSensor : EnvironmentSensor;

        port occupantEntry : OccupantPort;
        port fuelInlet : FuelPort;
        port navDataPort : DataPort;
        port envDataPort : DataPort;
    }

    part def TransportationScenario {
        attribute scenarioName : String;
        attribute journeyCompleted : Boolean;
        attribute safetyEnsured : Boolean;
        attribute sustainabilityConsidered : Boolean;

        part vehicle : Vehicle;
        part environment : Infrastructure;
        part refuelStation : GasStation;

        port scenarioOccupantPort : OccupantPort;
        port scenarioFuelPort : FuelPort;
        port scenarioNavPort : DataPort;
        port scenarioEnvPort : DataPort;

        connect vehicle.occupantEntry to scenarioOccupantPort;
        connect vehicle.fuelInlet to scenarioFuelPort;
        connect vehicle.navDataPort to scenarioNavPort;
        connect vehicle.envDataPort to scenarioEnvPort;
    }
}