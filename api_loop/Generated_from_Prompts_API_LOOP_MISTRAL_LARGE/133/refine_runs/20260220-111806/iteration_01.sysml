package AutomotiveECUModel {
    private import ScalarValues::*;

    enum def CommunicationProtocol {
        enum CAN
    }

    part def VehicleControlUnit;
    part def EngineControlUnit;
    part def CANBus;
    part def BusInterface;

    item def SensorData;

    port def DataPort {
        attribute protocol : CommunicationProtocol = CommunicationProtocol::CAN;
    }

    part vehicle : VehicleControlUnit {
        port busIF : DataPort;
    }

    part engine : EngineControlUnit {
        port busIF : DataPort;
    }

    part canBus : CANBus {
        port vcuPort : DataPort;
        port ecuPort : DataPort;
    }

    part vcuBusIF : BusInterface {
        port internalPort : DataPort;
        port externalPort : DataPort;
    }

    part ecuBusIF : BusInterface {
        port internalPort : DataPort;
        port externalPort : DataPort;
    }

    connect vehicle.busIF to vcuBusIF.internalPort;
    connect vcuBusIF.externalPort to canBus.vcuPort;
    connect engine.busIF to ecuBusIF.internalPort;
    connect ecuBusIF.externalPort to canBus.ecuPort;
}