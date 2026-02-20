package SystemModel {
    private import ScalarValues::*;

    part def SystemAssembly;

    part def ComputerSubsystem;
    part def StorageSubsystem;

    part def SoftwareDesign {
        attribute messageFormat : String;
        attribute dataFormat : String;
    };

    part systemAssembly : SystemAssembly {
        part computer : ComputerSubsystem;
        part storage : StorageSubsystem;
        part software : SoftwareDesign;
    };

    item def Message;
    item def Data;

    port def MessagePort;
    port def DataPort;

    part computer : ComputerSubsystem {
        port messagePort : MessagePort;
        port dataPort : DataPort;
    };

    part storage : StorageSubsystem {
        port messagePort : MessagePort;
        port dataPort : DataPort;
    };

    connect systemAssembly.computer.messagePort to systemAssembly.storage.messagePort;
    connect systemAssembly.computer.dataPort to systemAssembly.storage.dataPort;
}