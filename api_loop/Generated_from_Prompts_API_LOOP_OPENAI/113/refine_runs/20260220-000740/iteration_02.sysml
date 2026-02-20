package VehicleSystem {
  private import ScalarValues::*;

  enum def VehicleState {
    enum Shutdown;
    enum Starting;
    enum Normal;
    enum Degraded;
    enum Maintenance;
    enum Fault;
  }

  item def StartSignal {
    attribute value : Boolean;
  }

  item def BrakeSignal {
    attribute value : Boolean;
  }

  item def TemperatureSignal {
    attribute value : Real;
  }

  item def WarningSignal {
    attribute value : Boolean;
  }

  item def StatusSignal {
    attribute stateValue : VehicleState;
  }

  part def Vehicle {
    item startIn : StartSignal;
    item brakeIn : BrakeSignal;
    item tempOut : TemperatureSignal;
    item warnOut : WarningSignal;
    item statusOut : StatusSignal;

    attribute currentState : VehicleState = VehicleState::Shutdown;
    attribute fuelLevel : Real;
    attribute maintenanceIntervalDays : Integer;
    attribute temperatureThreshold : Real;
    attribute temperature : Real;
    attribute brakePressed : Boolean;
    attribute startRequested : Boolean;
  }

  part def VehicleController {
    item startOut : StartSignal;
    item brakeOut : BrakeSignal;
    item tempIn : TemperatureSignal;
    item warnIn : WarningSignal;
    item statusIn : StatusSignal;

    attribute lastWarning : Boolean;
    attribute lastState : VehicleState;
    attribute startCommand : Boolean;
    attribute brakeCommand : Boolean;
  }

  part def VehicleSystem {
    attribute defaultFuelLevel : Real = 50.0;
    attribute defaultMaintenanceIntervalDays : Integer = 30;
    attribute defaultTemperatureThreshold : Real = 90.0;

    part vehicle : Vehicle {
      fuelLevel = defaultFuelLevel;
      maintenanceIntervalDays = defaultMaintenanceIntervalDays;
      temperatureThreshold = defaultTemperatureThreshold;
    }

    part controller : VehicleController;

    connect vehicle.startIn to controller.startOut;
    connect vehicle.brakeIn to controller.brakeOut;
    connect vehicle.tempOut to controller.tempIn;
    connect vehicle.warnOut to controller.warnIn;
    connect vehicle.statusOut to controller.statusIn;
  }
}