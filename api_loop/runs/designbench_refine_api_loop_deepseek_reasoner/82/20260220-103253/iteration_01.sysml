package SystemModel {
    private import ScalarValues::*;
    
    item def DataFormat {
        attribute id : Integer;
        attribute timestamp : String;
        attribute payload : String;
        attribute checksum : Integer;
    }
    
    item def MessageFormat {
        attribute source : String;
        attribute destination : String;
        attribute messageType : String;
        attribute priority : Integer;
        attribute data : DataFormat;
    }
    
    port def SoftwareInterface {
        attribute messagePort : MessageFormat;
        attribute dataPort : DataFormat;
    }
    
    part def ComputerSubsystem {
        port softwareInterface : SoftwareInterface;
        attribute processor : String;
        attribute memory : Integer;
    }
    
    part def StorageSubsystem {
        port softwareInterface : SoftwareInterface;
        attribute capacity : Integer;
        attribute type : String;
    }
    
    part def SystemAssembly {
        part computer : ComputerSubsystem;
        part storage : StorageSubsystem;
        
        connect computer.softwareInterface to storage.softwareInterface;
    }
}