package SystemModel {
    private import ScalarValues::*;

    part def SystemAssembly;
    part def Computer;
    part def Storage;
    part def SoftwareDesign;

    item def MessageFormat;
    item def DataFormat;

    port def MessagePort;
    port def DataPort;
    port def SoftwarePort;

    part systemAssembly : SystemAssembly {
        part computer : Computer;
        part storage : Storage;
        part software : SoftwareDesign;
    }

    part def SoftwareDesign {
        attribute messageFormat : MessageFormat;
        attribute dataFormat : DataFormat;
        port softwarePort : SoftwarePort;
    }

    part def Computer {
        attribute canUtilizeFunctions : Boolean = true;
        port softwarePort : SoftwarePort;
    }

    part def Storage {
        attribute canAccessMessageFormat : Boolean = true;
        attribute isCompatibleWithDataFormat : Boolean = true;
        port messagePort : MessagePort;
        port dataPort : DataPort;
    }

    connect systemAssembly.software.softwarePort to systemAssembly.computer.softwarePort;
    connect systemAssembly.software::messageFormat to systemAssembly.storage.messagePort;
    connect systemAssembly.software::dataFormat to systemAssembly.storage.dataPort;
}