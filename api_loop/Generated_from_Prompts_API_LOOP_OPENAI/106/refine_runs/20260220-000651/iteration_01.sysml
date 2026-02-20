package VehiclePowertrainManagement {
    private import ScalarValues::*;
    
    item def FuelCommand;
    item def StartSignal;
    item def ShutdownSignal;
    item def Torque;
    
    enum def EngineState {
        Off;
        On;
    }
    
    port def FuelCommandPort {
        item fuelCmd : FuelCommand;
    }
    
    port def StartSignalPort {
        item startSig : StartSignal;
    }
    
    port def ShutdownSignalPort {
        item shutdownSig : ShutdownSignal;
    }
    
    port def TorquePort {
        item torque : Torque;
    }
    
    part def DriverInput {
        port fuelOut : FuelCommandPort;
        port startOut : StartSignalPort;
        port shutdownOut : ShutdownSignalPort;
    }
    
    part def Engine {
        port fuelIn : FuelCommandPort;
        port startIn : StartSignalPort;
        port shutdownIn : ShutdownSignalPort;
        port torqueOut : TorquePort;
        attribute state : EngineState = Off;
    }
    
    part def Transmission {
        port torqueIn : TorquePort;
        port torqueOut : TorquePort;
        attribute gearRatio : Real = 1.0;
    }
    
    part def TorqueDistributor {
        port torqueIn : TorquePort;
        port torqueOutFL : TorquePort;
        port torqueOutFR : TorquePort;
        port torqueOutRL : TorquePort;
        port torqueOutRR : TorquePort;
    }
    
    part def Wheel {
        port torqueIn : TorquePort;
        attribute wheelTorque : Torque;
    }
    
    part def PowertrainSystem {
        part driver : DriverInput;
        part engine : Engine;
        part transmission : Transmission;
        part distributor : TorqueDistributor;
        part wheelFL : Wheel;
        part wheelFR : Wheel;
        part wheelRL : Wheel;
        part wheelRR : Wheel;
        
        connect driver.fuelOut to engine.fuelIn;
        connect driver.startOut to engine.startIn;
        connect driver.shutdownOut to engine.shutdownIn;
        connect engine.torqueOut to transmission.torqueIn;
        connect transmission.torqueOut to distributor.torqueIn;
        connect distributor.torqueOutFL to wheelFL.torqueIn;
        connect distributor.torqueOutFR to wheelFR.torqueIn;
        connect distributor.torqueOutRL to wheelRL.torqueIn;
        connect distributor.torqueOutRR to wheelRR.torqueIn;
    }
    
    part powertrainSystem : PowertrainSystem;
}