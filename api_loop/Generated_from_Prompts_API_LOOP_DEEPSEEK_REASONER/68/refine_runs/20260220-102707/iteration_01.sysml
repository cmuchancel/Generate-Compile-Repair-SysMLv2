package VehicleSystem {
    private import ScalarValues::*;
    
    port def MechanicalPort;
    
    part def Engine {
        attribute speed : Real;
        attribute torque : Real;
        port output : MechanicalPort;
    }
    
    part def Transmission {
        attribute gearRatio : Real;
        port input : MechanicalPort;
        port output : MechanicalPort;
    }
    
    part def Vehicle {
        attribute curbWeight : Real;
        attribute fuelMass : Real;
        attribute maxMassLimit : Real;
        
        part engine : Engine;
        part transmission : Transmission;
        
        connect engine.output to transmission.input;
    }
}