package SystemModel {
    private import ScalarValues::*;

    part def SystemAssembly;
    part def Computer;
    part def Storage;
    part def Software;

    item def MessageFormat;
    item def DataFormat;

    part systemAssembly : SystemAssembly {
        part computer : Computer;
        part storage : Storage;
        part software : Software;
    }

    part def Software {
        attribute messageFormat : MessageFormat;
        attribute dataFormat : DataFormat;
    }

    part def Computer {
        attribute canIntegrateFunctions : Boolean = true;
    }

    part def Storage {
        attribute canAccessMessageFormat : Boolean = true;
        attribute isCompatibleWithDataFormat : Boolean = true;
    }

    connect systemAssembly.software.messageFormat to systemAssembly.storage::Storage;
    connect systemAssembly.software.dataFormat to systemAssembly.storage::Storage;
    connect systemAssembly.computer::Computer to systemAssembly.software::Software;
}