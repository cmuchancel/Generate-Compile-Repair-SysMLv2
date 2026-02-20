package SystemModel {
    private import ScalarValues::*;

    part def SystemAssembly;
    part def Computer;
    part def Storage;
    part def SoftwareDesign;

    item def MessageFormat;
    item def DataFormat;

    port def SoftwarePort;
    port def MessagePort;
    port def DataPort;

    part systemAssembly : SystemAssembly {
        part computer : Computer;
        part storage : Storage;
        part software : SoftwareDesign;
    }

    part def SoftwareDesign {
        attribute messageFormat : MessageFormat;
        attribute dataFormat : DataFormat;
        port softwarePort : SoftwarePort;
        port messagePort : MessagePort;
        port dataPort : DataPort;
    }

    part def Computer {
        attribute canIntegrateFunctions : Boolean = true;
        port softwarePort : SoftwarePort;
    }

    part def Storage {
        attribute canAccessMessageFormat : Boolean = true;
        attribute isCompatibleWithDataFormat : Boolean = true;
        port messagePort : MessagePort;
        port dataPort : DataPort;
    }

    connect systemAssembly.software.messagePort to systemAssembly.storage.messagePort;
    connect systemAssembly.software.dataPort to systemAssembly.storage.dataPort;
    connect systemAssembly.computer.softwarePort to systemAssembly.software.softwarePort;
}