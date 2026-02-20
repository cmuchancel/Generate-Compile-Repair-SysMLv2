package AutomotiveECU {
    private import ScalarValues::*;

    item def CANMessage;
    item def SensorData;

    port def CANPort {
        item ref msg: CANMessage;
    }

    port def SensorPort {
        item ref data: SensorData;
    }

    part def BusIF {
        port busPort: CANPort;
        port controllerPort: CANPort;
    }

    part def VCU {
        part busIF: BusIF;
        port canPort: CANPort;
        port sensorPort: SensorPort;
        connect canPort to busIF.controllerPort;
    }

    part def ECU {
        part busIF: BusIF;
        port canPort: CANPort;
        port sensorPort: SensorPort;
        connect canPort to busIF.controllerPort;
    }

    part def CANBus {
        port busPort: CANPort;
    }

    part def VehicleSystem {
        part vcu: VCU;
        part ecu: ECU;
        part canBus: CANBus;

        connect vcu.busIF.busPort to canBus.busPort;
        connect ecu.busIF.busPort to canBus.busPort;
    }
}