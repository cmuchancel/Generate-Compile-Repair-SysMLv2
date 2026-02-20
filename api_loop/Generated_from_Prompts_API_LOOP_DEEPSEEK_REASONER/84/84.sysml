package PowerTransmissionSystem {
    private import ScalarValues::*;
    
    part def LogicalSystem {
        part torqueGenerator : TorqueGenerator;
    }
    
    part def TorqueGenerator {
        attribute torqueGeneration : TorqueGenerationAction;
    }
    
    item def TorqueGenerationAction;
    
    part def PhysicalSystem {
        part powertrain : Powertrain;
    }
    
    part def Powertrain {
        part engine : Engine;
    }
    
    part def Engine {
        attribute torqueGeneration : TorqueGenerationAction;
    }
    
    part sys : LogicalSystem;
    part phys : PhysicalSystem;
    
    allocate sys.torqueGenerator to phys.powertrain;
    allocate sys.torqueGenerator.torqueGeneration to phys.powertrain.engine.torqueGeneration;
}