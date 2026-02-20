package SystemDecomposition {
    private import ScalarValues::*;

    enum def SystemStatus {
        enum NOMINAL;
        enum DEGRADED;
        enum FAULT;
    }

    enum def InterfaceType {
        enum ELECTRICAL;
        enum MECHANICAL;
        enum DATA;
        enum THERMAL;
    }

    item def PowerSignal {
        attribute voltage : Real;
        attribute current : Real;
    }

    item def DataSignal {
        attribute bandwidth : Real;
        attribute latency : Real;
    }

    item def ThermalSignal {
        attribute temperature : Real;
        attribute heatFlux : Real;
    }

    item def MechanicalSignal {
        attribute force : Real;
        attribute torque : Real;
    }

    port def PowerPort {
        attribute interfaceType : InterfaceType;
        attribute nominalVoltage : Real;
        attribute maxCurrent : Real;
    }

    port def DataPort {
        attribute interfaceType : InterfaceType;
        attribute protocol : String;
        attribute dataRate : Real;
    }

    port def ThermalPort {
        attribute interfaceType : InterfaceType;
        attribute maxTemperature : Real;
        attribute coolingCapacity : Real;
    }

    port def MechanicalPort {
        attribute interfaceType : InterfaceType;
        attribute maxForce : Real;
        attribute maxTorque : Real;
    }

    part def PowerSupplyUnit {
        attribute unitId : String;
        attribute outputVoltage : Real;
        attribute outputCurrent : Real;
        attribute efficiency : Real;
        attribute status : SystemStatus;
        port powerOut : PowerPort;
        port thermalOut : ThermalPort;
        port controlIn : DataPort;
    }

    part def ProcessingUnit {
        attribute unitId : String;
        attribute clockSpeed : Real;
        attribute memorySize : Real;
        attribute processingLoad : Real;
        attribute status : SystemStatus;
        port powerIn : PowerPort;
        port dataIn : DataPort;
        port dataOut : DataPort;
        port thermalOut : ThermalPort;
    }

    part def SensorUnit {
        attribute unitId : String;
        attribute sensorType : String;
        attribute samplingRate : Real;
        attribute accuracy : Real;
        attribute status : SystemStatus;
        port powerIn : PowerPort;
        port dataOut : DataPort;
        port thermalOut : ThermalPort;
        port mechanicalMount : MechanicalPort;
    }

    part def ActuatorUnit {
        attribute unitId : String;
        attribute actuatorType : String;
        attribute maxForce : Real;
        attribute responseTime : Real;
        attribute status : SystemStatus;
        port powerIn : PowerPort;
        port commandIn : DataPort;
        port mechanicalOut : MechanicalPort;
        port thermalOut : ThermalPort;
    }

    part def CommunicationUnit {
        attribute unitId : String;
        attribute protocol : String;
        attribute dataRate : Real;
        attribute range : Real;
        attribute status : SystemStatus;
        port powerIn : PowerPort;
        port internalData : DataPort;
        port externalData : DataPort;
        port thermalOut : ThermalPort;
    }

    part def ThermalManagementUnit {
        attribute unitId : String;
        attribute coolingCapacity : Real;
        attribute operatingTempMin : Real;
        attribute operatingTempMax : Real;
        attribute status : SystemStatus;
        port thermalIn : ThermalPort;
        port powerIn : PowerPort;
        port statusOut : DataPort;
    }

    part def StructuralUnit {
        attribute unitId : String;
        attribute material : String;
        attribute mass : Real;
        attribute loadCapacity : Real;
        port mechanicalIn : MechanicalPort;
        port mechanicalOut : MechanicalPort;
    }

    part def PowerSubsystem {
        attribute subsystemId : String;
        attribute totalPowerCapacity : Real;
        attribute status : SystemStatus;
        port mainPowerOut : PowerPort;
        port thermalOut : ThermalPort;
        port controlIn : DataPort;
        part primaryPSU : PowerSupplyUnit {
            attribute unitId = "PSU-001";
            attribute outputVoltage = 28.0;
            attribute outputCurrent = 50.0;
            attribute efficiency = 0.92;
            attribute status = SystemStatus::NOMINAL;
        }
        part redundantPSU : PowerSupplyUnit {
            attribute unitId = "PSU-002";
            attribute outputVoltage = 28.0;
            attribute outputCurrent = 50.0;
            attribute efficiency = 0.91;
            attribute status = SystemStatus::NOMINAL;
        }
    }

    part def ComputingSubsystem {
        attribute subsystemId : String;
        attribute status : SystemStatus;
        port powerIn : PowerPort;
        port sensorDataIn : DataPort;
        port commandOut : DataPort;
        port commLink : DataPort;
        port thermalOut : ThermalPort;
        part mainProcessor : ProcessingUnit {
            attribute unitId = "CPU-001";
            attribute clockSpeed = 2400.0;
            attribute memorySize = 16384.0;
            attribute processingLoad = 0.65;
            attribute status = SystemStatus::NOMINAL;
        }
        part coProcessor : ProcessingUnit {
            attribute unitId = "CPU-002";
            attribute clockSpeed = 1200.0;
            attribute memorySize = 8192.0;
            attribute processingLoad = 0.40;
            attribute status = SystemStatus::NOMINAL;
        }
    }

    part def SensorSubsystem {
        attribute subsystemId : String;
        attribute status : SystemStatus;
        port powerIn : PowerPort;
        port dataOut : DataPort;
        port thermalOut : ThermalPort;
        port mechanicalMount : MechanicalPort;
        part primarySensor : SensorUnit {
            attribute unitId = "SEN-001";
            attribute sensorType = "Inertial";
            attribute samplingRate = 1000.0;
            attribute accuracy = 0.001;
            attribute status = SystemStatus::NOMINAL;
        }
        part secondarySensor : SensorUnit {
            attribute unitId = "SEN-002";
            attribute sensorType = "Environmental";
            attribute samplingRate = 100.0;
            attribute accuracy = 0.01;
            attribute status = SystemStatus::NOMINAL;
        }
        part tertiarySensor : SensorUnit {
            attribute unitId = "SEN-003";
            attribute sensorType = "Optical";
            attribute samplingRate = 60.0;
            attribute accuracy = 0.0001;
            attribute status = SystemStatus::NOMINAL;
        }
    }

    part def ActuatorSubsystem {
        attribute subsystemId : String;
        attribute status : SystemStatus;
        port powerIn : PowerPort;
        port commandIn : DataPort;
        port mechanicalOut : MechanicalPort;
        port thermalOut : ThermalPort;
        part primaryActuator : ActuatorUnit {
            attribute unitId = "ACT-001";
            attribute actuatorType = "Linear";
            attribute maxForce = 5000.0;
            attribute responseTime = 0.05;
            attribute status = SystemStatus::NOMINAL;
        }
        part secondaryActuator : ActuatorUnit {
            attribute unitId = "ACT-002";
            attribute actuatorType = "Rotary";
            attribute maxForce = 3000.0;
            attribute responseTime = 0.08;
            attribute status = SystemStatus::NOMINAL;
        }
    }

    part def CommunicationSubsystem {
        attribute subsystemId : String;
        attribute status : SystemStatus;
        port powerIn : PowerPort;
        port internalDataIn : DataPort;
        port externalDataOut : DataPort;
        port thermalOut : ThermalPort;
        part primaryComm : CommunicationUnit {
            attribute unitId = "COM-001";
            attribute protocol = "Ethernet";
            attribute dataRate = 1000.0;
            attribute range = 100.0;
            attribute status = SystemStatus::NOMINAL;
        }
        part redundantComm : CommunicationUnit {
            attribute unitId = "COM-002";
            attribute protocol = "CAN";
            attribute dataRate = 1.0;
            attribute range = 40.0;
            attribute status = SystemStatus::NOMINAL;
        }
    }

    part def ThermalSubsystem {
        attribute subsystemId : String;
        attribute status : SystemStatus;
        port thermalIn : ThermalPort;
        port powerIn : PowerPort;
        port statusOut : DataPort;
        part primaryThermal : ThermalManagementUnit {
            attribute unitId = "THM-001";
            attribute coolingCapacity = 2000.0;
            attribute operatingTempMin = -40.0;
            attribute operatingTempMax = 85.0;
            attribute status = SystemStatus::NOMINAL;
        }
    }

    part def StructuralSubsystem {
        attribute subsystemId : String;
        port mechanicalIn : MechanicalPort;
        port mechanicalOut : MechanicalPort;
        part primaryStructure : StructuralUnit {
            attribute unitId = "STR-001";
            attribute material = "AluminumAlloy";
            attribute mass = 120.0;
            attribute loadCapacity = 50000.0;
        }
        part secondaryStructure : StructuralUnit {
            attribute unitId = "STR-002";
            attribute material = "CarbonFiber";
            attribute mass = 45.0;
            attribute loadCapacity = 30000.0;
        }
    }

    part def TopLevelSystem {
        attribute systemId : String;
        attribute systemName : String;
        attribute missionProfile : String;
        attribute overallStatus : SystemStatus;
        attribute designLifetime : Real;
        attribute operatingAltitude : Real;

        part powerSub : PowerSubsystem {
            attribute subsystemId = "PWR-SUB-001";
            attribute totalPowerCapacity = 2800.0;
            attribute status = SystemStatus::NOMINAL;
        }

        part computingSub : ComputingSubsystem {
            attribute subsystemId = "COMP-SUB-001";
            attribute status = SystemStatus::NOMINAL;
        }

        part sensorSub : SensorSubsystem {
            attribute subsystemId = "SEN-SUB-001";
            attribute status = SystemStatus::NOMINAL;
        }

        part actuatorSub : ActuatorSubsystem {
            attribute subsystemId = "ACT-SUB-001";
            attribute status = SystemStatus::NOMINAL;
        }

        part commSub : CommunicationSubsystem {
            attribute subsystemId = "COM-SUB-001";
            attribute status = SystemStatus::NOMINAL;
        }

        part thermalSub : ThermalSubsystem {
            attribute subsystemId = "THM-SUB-001";
            attribute status = SystemStatus::NOMINAL;
        }

        part structuralSub : StructuralSubsystem {
            attribute subsystemId = "STR-SUB-001";
        }

        connect powerSub.mainPowerOut to computingSub.powerIn;
        connect powerSub.mainPowerOut to sensorSub.powerIn;
        connect powerSub.mainPowerOut to actuatorSub.powerIn;
        connect powerSub.mainPowerOut to commSub.powerIn;
        connect powerSub.mainPowerOut to thermalSub.powerIn;
        connect powerSub.thermalOut to thermalSub.thermalIn;
        connect sensorSub.dataOut to computingSub.sensorDataIn;
        connect computingSub.commandOut to actuatorSub.commandIn;
        connect computingSub.commLink to commSub.internalDataIn;
        connect computingSub.thermalOut to thermalSub.thermalIn;
        connect sensorSub.thermalOut to thermalSub.thermalIn;
        connect actuatorSub.thermalOut to thermalSub.thermalIn;
        connect commSub.thermalOut to thermalSub.thermalIn;
        connect thermalSub.statusOut to computingSub.sensorDataIn;
        connect sensorSub.mechanicalMount to structuralSub.mechanicalOut;
        connect actuatorSub.mechanicalOut to structuralSub.mechanicalIn;
    }

    part systemInstance : TopLevelSystem {
        attribute systemId = "SYS-001";
        attribute systemName = "IntegratedMissionSystem";
        attribute missionProfile = "AutonomousOperation";
        attribute overallStatus = SystemStatus::NOMINAL;
        attribute designLifetime = 10.0;
        attribute operatingAltitude = 500.0;
    }
}