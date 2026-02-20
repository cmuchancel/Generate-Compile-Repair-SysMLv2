package AutomotiveECU {
    private import ScalarValues::*;

    enum def NetworkStatus {
        enum ACTIVE;
        enum INACTIVE;
        enum ERROR;
    }

    enum def ControlMode {
        enum NORMAL;
        enum DIAGNOSTIC;
        enum SAFE;
    }

    item def CANMessage {
        attribute messageId : Integer;
        attribute payload : Real;
        attribute timestamp : Real;
    }

    item def SensorData {
        attribute sensorId : Integer;
        attribute value : Real;
        attribute unit : String;
    }

    port def CANPort {
        attribute busSpeed : Integer;
        attribute nodeId : Integer;
    }

    port def SensorPort {
        attribute channelId : Integer;
        attribute sampleRate : Real;
    }

    part def BusInterface {
        attribute interfaceId : Integer;
        attribute busAddress : Integer;
        attribute status : NetworkStatus;
        port canConnection : CANPort;
        port sensorConnection : SensorPort;
    }

    part def CANBus {
        attribute busId : Integer;
        attribute baudRate : Integer;
        attribute networkStatus : NetworkStatus;
        attribute maxNodes : Integer;
        port vcuBusPort : CANPort;
        port ecuBusPort : CANPort;
    }

    part def VehicleControlUnit {
        attribute vcuId : Integer;
        attribute firmwareVersion : String;
        attribute controlMode : ControlMode;
        attribute vehicleSpeed : Real;
        attribute batteryVoltage : Real;
        attribute operationalStatus : NetworkStatus;
        port vcuCANPort : CANPort;
        port vcuSensorPort : SensorPort;
        part vcuBusIF : BusInterface;
    }

    part def EngineControlUnit {
        attribute ecuId : Integer;
        attribute firmwareVersion : String;
        attribute controlMode : ControlMode;
        attribute engineRPM : Real;
        attribute engineTemp : Real;
        attribute throttlePosition : Real;
        attribute operationalStatus : NetworkStatus;
        port ecuCANPort : CANPort;
        port ecuSensorPort : SensorPort;
        part ecuBusIF : BusInterface;
    }

    part def AutomotiveSystem {
        attribute systemId : Integer;
        attribute systemName : String;

        part vcu : VehicleControlUnit;
        part ecu : EngineControlUnit;
        part canBus : CANBus;

        connect vcu.vcuCANPort to canBus.vcuBusPort;
        connect ecu.ecuCANPort to canBus.ecuBusPort;
        connect vcu.vcuBusIF.canConnection to vcu.vcuCANPort;
        connect ecu.ecuBusIF.canConnection to ecu.ecuCANPort;
        connect vcu.vcuSensorPort to ecu.ecuSensorPort;
    }
}