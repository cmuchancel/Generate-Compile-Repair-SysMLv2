package VehicleSystem {
  private import ScalarValues::*;
  
  enum def VehicleState {
    enum Off;
    enum Starting;
    enum Running;
    enum HealthManagement;
    enum Maintenance;
    enum Degraded;
    enum Normal;
  }
  
  enum def ControllerState {
    enum Off;
    enum Starting;
    enum Running;
    enum Shutdown;
  }
  
  item def SignalItem {
    attribute value : Boolean;
  }
  
  port def SignalPort {
    item signal : SignalItem;
  }
  
  part def VehicleA {
    attribute currentState : VehicleState = Off;
    attribute brakePressed : Boolean;
    attribute startSignal : Boolean;
    attribute vehicleOnSignal : Boolean;
    attribute temperature : Real;
    attribute tempThreshold : Real = 100.0;
    attribute overTemp : Boolean;
    attribute degradedMode : Boolean;
    attribute restoreNormalSignal : Boolean;
    attribute parkingBrakeApplied : Boolean;
    attribute powerSupplied : Boolean;
    attribute selfCheckCompleted : Boolean;
    
    port startPort : SignalPort;
    port shutdownPort : SignalPort;
    port overTempPort : SignalPort;
    port restorePort : SignalPort;
  }
  
  part def VehicleController {
    attribute controllerState : ControllerState = Off;
    attribute startupSignal : Boolean;
    attribute shutdownSignal : Boolean;
    
    port startPort : SignalPort;
    port shutdownPort : SignalPort;
    port overTempPort : SignalPort;
    port restorePort : SignalPort;
  }
  
  part vehicleA : VehicleA;
  part controller : VehicleController;
  
  connect vehicleA.startPort to controller.startPort;
  connect vehicleA.shutdownPort to controller.shutdownPort;
  connect vehicleA.overTempPort to controller.overTempPort;
  connect vehicleA.restorePort to controller.restorePort;
}