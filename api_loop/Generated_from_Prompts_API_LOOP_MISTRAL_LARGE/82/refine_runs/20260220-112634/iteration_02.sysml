package SystemModel {
    private import ScalarValues::*;

    part def SystemAssembly {
        part computer : ComputerSubsystem;
        part storage : StorageSubsystem;
        part software : SoftwareDesign;
    }

    part def ComputerSubsystem {
        port messagePort : MessagePort;
        port dataPort : DataPort;
    }

    part def StorageSubsystem {
        port messagePort : MessagePort;
        port dataPort : DataPort;
    }

    part def SoftwareDesign {
        attribute messageFormat : String;
        attribute dataFormat : String;
    }

    item def Message;
    item def Data;

    port def MessagePort;
    port def DataPort;

    part systemAssembly : SystemAssembly;

    connect systemAssembly.computer.messagePort to systemAssembly.storage.messagePort;
    connect systemAssembly.computer.dataPort to systemAssembly.storage.dataPort;
}