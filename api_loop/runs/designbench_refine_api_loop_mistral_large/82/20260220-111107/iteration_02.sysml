package SystemModel {
    private import ScalarValues::*;

    part def SystemAssembly;
    part def ComputerSubsystem;
    part def StorageSubsystem;
    part def SoftwareDesignSection;

    item def MessageFormat;
    item def DataFormat;

    part systemAssembly : SystemAssembly {
        part computer : ComputerSubsystem;
        part storage : StorageSubsystem;
        part software : SoftwareDesignSection;
    }

    part def SoftwareDesignSection {
        attribute messageFormat : MessageFormat;
        attribute dataFormat : DataFormat;
    }

    part def ComputerSubsystem {
        attribute canUtilizeFunctions : Boolean = true;
        port softwarePort;
    }

    part def StorageSubsystem {
        attribute canAccessMessageFormat : Boolean = true;
        attribute isCompatibleWithDataFormat : Boolean = true;
        port messagePort;
        port dataPort;
    }

    connect systemAssembly.software.messageFormat to systemAssembly.storage.messagePort;
    connect systemAssembly.software.dataFormat to systemAssembly.storage.dataPort;
    connect systemAssembly.computer.softwarePort to systemAssembly.software;
}