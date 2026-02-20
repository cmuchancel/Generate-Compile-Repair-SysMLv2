package SystemModel {
    private import ScalarValues::*;

    part def SystemAssembly;
    part def ComputerSubsystem;
    part def StorageSubsystem;
    part def SoftwareDesignSection;

    item def MessageFormat;
    item def DataFormat;

    port def SoftwarePort;
    port def MessagePort;
    port def DataPort;

    part systemAssembly : SystemAssembly {
        part computer : ComputerSubsystem;
        part storage : StorageSubsystem;
        part software : SoftwareDesignSection;
    }

    part def SoftwareDesignSection {
        attribute messageFormat : MessageFormat;
        attribute dataFormat : DataFormat;
        port softwarePort : SoftwarePort;
    }

    part def ComputerSubsystem {
        attribute canUtilizeFunctions : Boolean = true;
        port softwarePort : SoftwarePort;
    }

    part def StorageSubsystem {
        attribute canAccessMessageFormat : Boolean = true;
        attribute isCompatibleWithDataFormat : Boolean = true;
        port messagePort : MessagePort;
        port dataPort : DataPort;
    }

    connect systemAssembly.software.softwarePort to systemAssembly.computer.softwarePort;
    connect systemAssembly.software.messageFormat to systemAssembly.storage.messagePort;
    connect systemAssembly.software.dataFormat to systemAssembly.storage.dataPort;
}