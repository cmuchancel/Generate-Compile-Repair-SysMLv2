package AutomobileAssembly {
    private import ScalarValues::*;

    enum def AssemblyStatus {
        enum Incomplete;
        enum Complete;
    }

    item def GearboxUnit {
        attribute functional : Boolean;
    }

    item def EngineUnit {
        attribute functional : Boolean;
    }

    part def VehicleWithoutGearboxAndEngine {
        attribute status : AssemblyStatus;
    }

    part def VehicleWithGearboxNoEngine {
        attribute status : AssemblyStatus;
        part gearbox : GearboxUnit;
    }

    part def FullyAssembledVehicle {
        attribute status : AssemblyStatus;
        part gearbox : GearboxUnit;
        part engine : EngineUnit;
    }

    port def AssemblyInputPort {
        attribute stageId : String;
    }

    port def AssemblyOutputPort {
        attribute stageId : String;
    }

    part def GearboxInstallationStep {
        attribute stepName : String;
        port input : AssemblyInputPort;
        port output : AssemblyOutputPort;
        part inputVehicle : VehicleWithoutGearboxAndEngine;
        part gearboxItem : GearboxUnit;
        part outputVehicle : VehicleWithGearboxNoEngine;
    }

    part def EngineInstallationStep {
        attribute stepName : String;
        port input : AssemblyInputPort;
        port output : AssemblyOutputPort;
        part inputVehicle : VehicleWithGearboxNoEngine;
        part engineItem : EngineUnit;
        part outputVehicle : FullyAssembledVehicle;
    }

    part def FinalAssemblyLine {
        attribute lineName : String;

        part step1 : GearboxInstallationStep {
            attribute stepName = "GearboxInstallation";
        }

        part step2 : EngineInstallationStep {
            attribute stepName = "EngineInstallation";
        }

        connect step1.output to step2.input;

        part finalVehicle : FullyAssembledVehicle {
            attribute status = AssemblyStatus::Complete;
            part gearbox : GearboxUnit {
                attribute functional = true;
            }
            part engine : EngineUnit {
                attribute functional = true;
            }
        }
    }

    part assemblyLine : FinalAssemblyLine {
        attribute lineName = "MainFinalAssemblyLine";
    }
}