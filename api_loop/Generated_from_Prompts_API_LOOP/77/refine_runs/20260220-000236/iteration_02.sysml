package TransportationSystem {
  private import ScalarValues::*;

  enum def DoorState { 
    enum Closed;
    enum Open;
  }

  port def DoorPort;
  port def VehiclePort;

  part def Person {
    attribute name : String;
  }

  part def Driver {
    attribute name : String;
    attribute licensed : Boolean;
  }

  part def Passenger {
    attribute name : String;
  }

  part def Seat {
    attribute occupied : Boolean;
    attribute comfortLevel : Real;
  }

  part def Door {
    port doorPort : DoorPort;
    attribute doorState : DoorState;
  }

  part def Environment {
    attribute weatherSafe : Boolean;
    attribute sustainabilityConsidered : Boolean;
  }

  part def Infrastructure {
    attribute gasStationCompatible : Boolean;
  }

  part def VehicleSystem {
    attribute capacity : Integer = 5;
    attribute passengerCapacity : Integer = 4;
    attribute safetyEnabled : Boolean;
    attribute comfortEnabled : Boolean;
    attribute easyEntryExit : Boolean;
    attribute humanCentered : Boolean;
    attribute supportsTrip : Boolean;

    port vehiclePort : VehiclePort;

    part driver : Driver;
    part passenger1 : Passenger;
    part passenger2 : Passenger;
    part passenger3 : Passenger;
    part passenger4 : Passenger;

    part driverSeat : Seat;
    part seat1 : Seat;
    part seat2 : Seat;
    part seat3 : Seat;
    part seat4 : Seat;

    part door1 : Door;
    part door2 : Door;
    part door3 : Door;
    part door4 : Door;

    part environment : Environment;
    part infrastructure : Infrastructure;

    connect door1.doorPort to vehiclePort;
    connect door2.doorPort to vehiclePort;
    connect door3.doorPort to vehiclePort;
    connect door4.doorPort to vehiclePort;
  }
}