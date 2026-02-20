package VehiclePowertrainSystem {
    private import ScalarValues::*;
    
    item def FuelCommand {
        attribute value : Real;
    }
    
    item def Torque {
        attribute value : Real;
    }
    
    enum def EngineState {
        enum Off;
        enum Running;
    }
    
    port def MechanicalInterface {
        attribute torque : Torque;
    }
    
    part def Engine {
        port in fuelInput : FuelCommand;
        port out torqueOutput : MechanicalInterface;
        attribute state : EngineState = EngineState::Off;
    }
    
    part def Transmission {
        port in inputShaft : MechanicalInterface;
        port out outputShaft : MechanicalInterface;
        attribute gearRatio : Real = 1.0;
    }
    
    part def Differential {
        port in input : MechanicalInterface;
        port out frontLeft : MechanicalInterface;
        port out frontRight : MechanicalInterface;
        port out rearLeft : MechanicalInterface;
        port out rearRight : MechanicalInterface;
    }
    
    part def Wheel {
        port in axle : MechanicalInterface;
    }
    
    part def VehiclePowertrain {
        part engine : Engine;
        part transmission : Transmission;
        part differential : Differential;
        part wheelFL : Wheel;
        part wheelFR : Wheel;
        part wheelRL : Wheel;
        part wheelRR : Wheel;
        
        connect engine.torqueOutput to transmission.inputShaft;
        connect transmission.outputShaft to differential.input;
        connect differential.frontLeft to wheelFL.axle;
        connect differential.frontRight to wheelFR.axle;
        connect differential.rearLeft to wheelRL.axle;
        connect differential.rearRight to wheelRR.axle;
    }
}