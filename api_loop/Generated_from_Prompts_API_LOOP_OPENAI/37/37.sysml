package VehiclePowerManagement {
  private import ScalarValues::*;

  enum def PowerState {
    enum Off;
    enum StartPhase;
    enum PoweredOn;
  }

  port def ControlPort {
    attribute startCommand : Boolean;
    attribute startedSignal : Boolean;
    attribute shutdownCommand : Boolean;
  }

  port def PowerPort {
    attribute supplyOn : Boolean;
  }

  part def User {
    port controlOut : ControlPort;
  }

  part def FunctionalUnit {
    port powerIn : PowerPort;
    attribute unitName : String;
  }

  part def VehiclePowerSystem {
    port controlIn : ControlPort;
    port powerOut : PowerPort;
    attribute powerState : PowerState = PowerState::Off;
    attribute selfDiagnosisActive : Boolean = false;
    attribute parkingBrakeEngaged : Boolean = false;
    attribute powerSupplyOn : Boolean = false;
  }

  part def Vehicle {
    part powerSystem : VehiclePowerSystem;
    part propulsionUnit : FunctionalUnit;
    part infotainmentUnit : FunctionalUnit;
    part safetyUnit : FunctionalUnit;

    connect powerSystem.powerOut to propulsionUnit.powerIn;
    connect powerSystem.powerOut to infotainmentUnit.powerIn;
    connect powerSystem.powerOut to safetyUnit.powerIn;
  }

  part def VehiclePowerManagementSystem {
    part user : User;
    part vehicle : Vehicle;

    connect user.controlOut to vehicle.powerSystem.controlIn;
  }
}