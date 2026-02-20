package AutomobileAssembly
  private import ScalarValues::*;

  part def Gearbox;
  part def Engine;

  item def VehicleWithoutGearboxEngine;
  item def GearboxItem;
  item def EngineItem;

  item def VehicleWithGearbox
    part gearbox : Gearbox;
  end;

  item def FullyAssembledVehicle
    part gearbox : Gearbox;
    part engine : Engine;
  end;

  port def VehicleWithoutGearboxIn
    item vehicle : VehicleWithoutGearboxEngine;
  end;

  port def GearboxIn
    item gearbox : GearboxItem;
  end;

  port def VehicleWithGearboxIn
    item vehicle : VehicleWithGearbox;
  end;

  port def VehicleWithGearboxOut
    item vehicle : VehicleWithGearbox;
  end;

  port def EngineIn
    item engine : EngineItem;
  end;

  port def FullyAssembledVehicleOut
    item vehicle : FullyAssembledVehicle;
  end;

  part def GearboxInstallation
    port inVehicle : VehicleWithoutGearboxIn;
    port inGearbox : GearboxIn;
    port outVehicle : VehicleWithGearboxOut;
  end;

  part def EngineInstallation
    port inVehicle : VehicleWithGearboxIn;
    port inEngine : EngineIn;
    port outVehicle : FullyAssembledVehicleOut;
  end;

  part def FinalAssemblyLine
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
  end;
end;