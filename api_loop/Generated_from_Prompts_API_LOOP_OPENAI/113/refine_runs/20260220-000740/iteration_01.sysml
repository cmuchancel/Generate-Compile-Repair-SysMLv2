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
    attribute state : VehicleState;
  }

  port def VehiclePort {
    item startIn : StartSignal;
    item brakeIn : BrakeSignal;
    item tempOut : TemperatureSignal;
    item warnOut : WarningSignal;
    item statusOut : StatusSignal;
  }

  port def ControllerPort {
    item startOut : StartSignal;
    item brakeOut : BrakeSignal;
    item tempIn : TemperatureSignal;
    item warnIn : WarningSignal;
    item statusIn : StatusSignal;
  }

  part def Vehicle {
    port vPort : VehiclePort;
    attribute currentState : VehicleState = VehicleState::Shutdown;
    attribute fuelLevel : Real;
    attribute maintenanceIntervalDays : Integer;
    attribute temperatureThreshold : Real;
    attribute temperature : Real;
    attribute brakePressed : Boolean;
    attribute startRequested : Boolean;
  }

  part def VehicleController {
    port cPort : ControllerPort;
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
      attribute fuelLevel = defaultFuelLevel;
      attribute maintenanceIntervalDays = defaultMaintenanceIntervalDays;
      attribute temperatureThreshold = defaultTemperatureThreshold;
    }

    part controller : VehicleController;

    connect vehicle.vPort.startIn to controller.cPort.startOut;
    connect vehicle.vPort.brakeIn to controller.cPort.brakeOut;
    connect vehicle.vPort.tempOut to controller.cPort.tempIn;
    connect vehicle.vPort.warnOut to controller.cPort.warnIn;
    connect vehicle.vPort.statusOut to controller.cPort.statusIn;
  }
}