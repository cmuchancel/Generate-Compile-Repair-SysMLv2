package SystemModel {
    private import ScalarValues::*;

    item def MessageFormat {
        attribute header : string;
        attribute payload : string;
        attribute timestamp : string;
        attribute checksum : integer;
    }

    item def DataFormat {
        attribute id : integer;
        attribute size : integer;
        attribute encodedData : string;
        attribute metadata : string;
    }

    port def DataPort {
        attribute msg : MessageFormat;
        attribute data : DataFormat;
    }

    part def ComputerSubsystem {
        port primaryInterface : DataPort;
        attribute processor : string;
    }

    part def StorageSubsystem {
        port storageInterface : DataPort;
        attribute capacity : integer;
    }

    part def SystemAssembly {
        part computer : ComputerSubsystem;
        part storage : StorageSubsystem;
        connect computer.primaryInterface to storage.storageInterface;
    }
}