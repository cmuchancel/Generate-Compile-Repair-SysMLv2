package SystemAssembly {
    private import ScalarValues::*;

    enum def MessageFormatType {
        enum JSON;
        enum XML;
        enum Binary;
    }

    enum def DataFormatType {
        enum Structured;
        enum Unstructured;
        enum SemiStructured;
    }

    enum def StorageAccessMode {
        enum ReadOnly;
        enum WriteOnly;
        enum ReadWrite;
    }

    item def MessageFormat {
        attribute formatType : MessageFormatType;
        attribute version : String;
        attribute encoding : String;
    }

    item def DataFormat {
        attribute dataType : DataFormatType;
        attribute schema : String;
        attribute compression : String;
    }

    port def DataPort {
        attribute transferRate : Real;
        attribute protocol : String;
    }

    port def StoragePort {
        attribute accessMode : StorageAccessMode;
        attribute capacity : Real;
    }

    port def SoftwareInterfacePort {
        attribute apiVersion : String;
        attribute supported : String;
    }

    part def SoftwareDesign {
        attribute designVersion : String;
        attribute messageFormat : MessageFormatType;
        attribute dataFormat : DataFormatType;

        part messageFormatSpec : MessageFormat;
        part dataFormatSpec : DataFormat;

        port softwareInterface : SoftwareInterfacePort;
    }

    part def ComputerSubsystem {
        attribute processorSpeed : Real;
        attribute memorySize : Real;
        attribute supportedMessageFormat : MessageFormatType;
        attribute supportedDataFormat : DataFormatType;
        attribute integratesSoftwareFunctions : Boolean;

        port dataOut : DataPort;
        port softwareLink : SoftwareInterfacePort;
    }

    part def StorageSubsystem {
        attribute storageCapacity : Real;
        attribute compatibleMessageFormat : MessageFormatType;
        attribute compatibleDataFormat : DataFormatType;
        attribute accessMode : StorageAccessMode;

        port storageAccess : StoragePort;
        port softwareLink : SoftwareInterfacePort;
    }

    part def SystemRoot {
        attribute systemName : String;
        attribute systemVersion : String;
        attribute reliable : Boolean;
        attribute efficient : Boolean;
        attribute compliant : Boolean;

        part software : SoftwareDesign;
        part computer : ComputerSubsystem;
        part storage : StorageSubsystem;

        connect computer.dataOut to storage.storageAccess;
        connect computer.softwareLink to software.softwareInterface;
        connect storage.softwareLink to software.softwareInterface;
    }

    part system : SystemRoot {
        attribute systemName = "IntegratedSystem";
        attribute systemVersion = "1.0";
        attribute reliable = true;
        attribute efficient = true;
        attribute compliant = true;

        part software : SoftwareDesign {
            attribute designVersion = "1.0";
            attribute messageFormat = MessageFormatType::JSON;
            attribute dataFormat = DataFormatType::Structured;
        }

        part computer : ComputerSubsystem {
            attribute processorSpeed = 3.5;
            attribute memorySize = 16.0;
            attribute supportedMessageFormat = MessageFormatType::JSON;
            attribute supportedDataFormat = DataFormatType::Structured;
            attribute integratesSoftwareFunctions = true;
        }

        part storage : StorageSubsystem {
            attribute storageCapacity = 1000.0;
            attribute compatibleMessageFormat = MessageFormatType::JSON;
            attribute compatibleDataFormat = DataFormatType::Structured;
            attribute accessMode = StorageAccessMode::ReadWrite;
        }

        connect computer.dataOut to storage.storageAccess;
        connect computer.softwareLink to software.softwareInterface;
        connect storage.softwareLink to software.softwareInterface;
    }
}