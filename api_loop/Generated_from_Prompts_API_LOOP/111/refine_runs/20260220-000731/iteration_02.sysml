package AutomobileAssembly {
  private import ScalarValues::*;

  part def Gearbox;
  part def Engine;

  part def VehicleWithoutGearboxEngine;
  part def VehicleWithGearbox {
    part gearbox : Gearbox;
  }
  part def FullyAssembledVehicle {
    part gearbox : Gearbox;
    part engine : Engine;
  }

  port def VehicleWithoutGearboxIn {
    attribute vehicle : VehicleWithoutGearboxEngine;
  }
  port def GearboxIn {
    attribute gearbox : Gearbox;
  }
  port def VehicleWithGearboxIn {
    attribute vehicle : VehicleWithGearbox;
  }
  port def VehicleWithGearboxOut {
    attribute vehicle : VehicleWithGearbox;
  }
  port def EngineIn {
    attribute engine : Engine;
  }
  port def FullyAssembledVehicleOut {
    attribute vehicle : FullyAssembledVehicle;
  }

  part def GearboxInstallation {
    port inVehicle : VehicleWithoutGearboxIn;
    port inGearbox : GearboxIn;
    port outVehicle : VehicleWithGearboxOut;
  }

  part def EngineInstallation {
    port inVehicle : VehicleWithGearboxIn;
    port inEngine : EngineIn;
    port outVehicle : FullyAssembledVehicleOut;
  }

  part def FinalAssemblyLine {
    port inVehicle : VehicleWithoutGearboxIn;
    port inGearbox : GearboxIn;
    port inEngine : EngineIn;
    port outVehicle : FullyAssembledVehicleOut;

    part step1 : GearboxInstallation;
    part step2 : EngineInstallation;

    connect inVehicle to step1.inVehicle;
    connect inGearbox to step1.inGearbox;
    connect step1.outVehicle to step2.inVehicle;
    connect inEngine to step2.inEngine;
    connect step2.outVehicle to outVehicle;
  }
}