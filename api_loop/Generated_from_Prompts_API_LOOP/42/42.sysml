package VehicleStartStopSystemPkg {
  private import ScalarValues::*;

  enum def VehicleState {
    off;
    starting;
    operating;
  }

  item def Signal;

  part def Controller {
    attribute controllerReady : Boolean = true;
  }

  part def StartStopSystem {
    attribute vehicleState : VehicleState = VehicleState::off;
    attribute brakePedalDepressed : Boolean = false;
    attribute startSignal : Boolean = false;
    attribute powerOnSignal : Boolean = false;
    attribute stopSignal : Boolean = false;
    attribute selfCheckComplete : Boolean = false;
    attribute powerSupplyOn : Boolean = false;
    attribute parkingBrakeApplied : Boolean = true;
    attribute startSignalToController : Boolean = false;
    part controller : Controller;
  }

  part def Vehicle {
    part startStop : StartStopSystem;
  }
}