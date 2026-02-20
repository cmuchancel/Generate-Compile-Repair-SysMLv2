package SystemModel {
    private import ScalarValues::*;

    item def MessageFormat {
        attribute header : String;
        attribute payload : String;
        attribute timestamp : String;
        attribute checksum : Integer;
    }

    item def DataFormat {
        attribute id : Integer;
        attribute size : Integer;
        attribute encodedData : String;
        attribute meta : String;
    }

    port def DataPort {
        attribute msg : MessageFormat;
        attribute data : DataFormat;
    }

    part def ComputerSubsystem {
        port primaryInterface : DataPort;
        attribute processor : String;
    }

    part def StorageSubsystem {
        port storageInterface : DataPort;
        attribute capacity : Integer;
    }

    part def SystemAssembly {
        part computer : ComputerSubsystem;
        part storage : StorageSubsystem;
        connect computer.primaryInterface to storage.storageInterface;
    }
}