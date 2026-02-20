package AutomobileAssembly {
    private import ScalarValues::*;

    part def VehicleAssembly {
        attribute id : String;
        port chassis : MechanicalInterface;
    }

    part def Gearbox {
        attribute id : String;
        port powerOutput : MechanicalInterface;
    }

    part def Engine {
        attribute id : String;
        port powerOutput : MechanicalInterface;
    }

    part def VehicleWithGearbox {
        part assembly : VehicleAssembly;
        part gearbox : Gearbox;
        connect assembly.chassis to gearbox.powerOutput;
    }

    part def FullyAssembledVehicle {
        part assembly : VehicleAssembly;
        part gearbox : Gearbox;
        part engine : Engine;
        connect assembly.chassis to gearbox.powerOutput;
        connect gearbox.powerOutput to engine.powerOutput;
    }

    part def GearboxInstallation {
        port inputAssembly : VehicleAssembly;
        port inputGearbox : Gearbox;
        port outputVehicle : VehicleWithGearbox;
    }

    part def EngineInstallation {
        port inputVehicle : VehicleWithGearbox;
        port inputEngine : Engine;
        port outputVehicle : FullyAssembledVehicle;
    }

    part def FinalAssemblyLine {
        part gearboxStation : GearboxInstallation;
        part engineStation : EngineInstallation;
        connect gearboxStation.outputVehicle to engineStation.inputVehicle;
    }

    port def MechanicalInterface;
}