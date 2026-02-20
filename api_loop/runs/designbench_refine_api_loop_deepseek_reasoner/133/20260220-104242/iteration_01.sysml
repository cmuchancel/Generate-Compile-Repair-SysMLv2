package AutomotiveECU {
    private import ScalarValues::*;
    
    port def BusIF;
    port def SensorPort;
    
    part def CANBus {
        port busIf : BusIF;
    }
    
    part def VehicleControlUnit {
        port busIf : BusIF;
        port sensor : SensorPort;
        attribute vehicleSpeed : Real;
        attribute brakePressure : Real;
    }
    
    part def EngineControlUnit {
        port busIf : BusIF;
        port sensor : SensorPort;
        attribute engineRPM : Integer;
        attribute throttlePosition : Real;
    }
    
    part def VehicleSystem {
        part vcu : VehicleControlUnit;
        part ecu : EngineControlUnit;
        part canBus : CANBus;
        
        connect vcu.busIf to canBus.busIf;
        connect ecu.busIf to canBus.busIf;
    }
}