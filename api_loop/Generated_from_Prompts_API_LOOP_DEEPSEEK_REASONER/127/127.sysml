package PowertrainSystem {
    private import ScalarValues::*;
    
    item def Torque {
        attribute value : Real;
        attribute unit : String = "Nm";
    }
    
    part def TorqueGenerator {
        port output : MechanicalPort;
        attribute generatedTorque : Torque;
    }
    
    part def Engine {
        port crankshaft : MechanicalPort;
        attribute displacement : Real;
        attribute maxTorque : Torque;
        attribute currentTorque : Torque;
    }
    
    part def Powertrain {
        part torqueGenerator : TorqueGenerator;
        part engine : Engine;
        
        connect torqueGenerator.output to engine.crankshaft;
    }
    
    port def MechanicalPort;
}