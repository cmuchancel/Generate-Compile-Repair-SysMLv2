package SystemModel {
    private import ScalarValues::*;

    part def SystemAssembly;
    part def ComputerSubsystem;
    part def StorageSubsystem;
    part def SoftwareDesign;

    item def MessageFormat;
    item def DataFormat;

    part systemAssembly : SystemAssembly {
        part computer : ComputerSubsystem;
        part storage : StorageSubsystem;
        part software : SoftwareDesign;
    }

    part def SoftwareDesign {
        attribute messageFormat : MessageFormat;
        attribute dataFormat : DataFormat;
    }

    part def ComputerSubsystem {
        attribute canUtilizeFunctions : Boolean = true;
    }

    part def StorageSubsystem {
        attribute canAccessMessageFormat : Boolean = true;
        attribute isCompatibleWithDataFormat : Boolean = true;
    }

    connect systemAssembly.software.messageFormat to systemAssembly.storage;
    connect systemAssembly.software.dataFormat to systemAssembly.storage;
    connect systemAssembly.computer to systemAssembly.software;
}