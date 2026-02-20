package PowerTransmissionSystem {
    private import ScalarValues::*;

    part def TorqueGenerator {
        attribute torqueOutput : Real;
        attribute powerOutput : Real;
    }

    part def Powertrain {
        part engine : Engine;
    }

    part def Engine {
        attribute torqueOutput : Real;
        attribute powerOutput : Real;
    }

    part system {
        part logicalTorqueGenerator : TorqueGenerator;
        part physicalPowertrain : Powertrain;
    }

    requirement PowerTransmission {
        attribute description : String = "The system shall provide power and generate torque.";
    }

    requirement TorqueGeneration {
        attribute description : String = "The torque generator shall generate torque to enable power transfer and conversion.";
    }

    requirement EngineTorqueAllocation {
        attribute description : String = "The engine shall perform the function of generating torque within the powertrain.";
    }

    requirement AllocationRequirement {
        attribute description : String = "The logical torque generator shall be allocated to the physical powertrain, with torque generation allocated to the engine.";
    }

    // Allocations
    allocate system::logicalTorqueGenerator to system::physicalPowertrain;
    allocate system::logicalTorqueGenerator::torqueOutput to system::physicalPowertrain::engine::torqueOutput;
    allocate system::logicalTorqueGenerator::powerOutput to system::physicalPowertrain::engine::powerOutput;
}