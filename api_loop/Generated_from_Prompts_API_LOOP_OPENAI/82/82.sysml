package SystemModel {
  private import ScalarValues::*;

  item def MessageFormat {
    attribute header : String;
    attribute payload : String;
  }

  item def DataFormat {
    attribute schema : String;
    attribute encoding : String;
  }

  part def SoftwareDesign {
    part messageFormat : MessageFormat;
    part dataFormat : DataFormat;
    attribute processingFunction : String;
    attribute transmissionFunction : String;
    attribute storageFunction : String;
  }

  part def ComputerSubsystem {
    part software : SoftwareDesign;
    attribute functionsIntegrated : Boolean = true;
  }

  part def StorageSubsystem {
    part messageFormat : MessageFormat;
    part dataFormat : DataFormat;
    attribute compatibleWithFormats : Boolean = true;
  }

  part def SystemAssembly {
    part computer : ComputerSubsystem;
    part storage : StorageSubsystem;
    attribute efficient : Boolean = true;
    attribute reliable : Boolean = true;
    attribute compliant : Boolean = true;
    connect computer.software.messageFormat to storage.messageFormat;
    connect computer.software.dataFormat to storage.dataFormat;
  }
}