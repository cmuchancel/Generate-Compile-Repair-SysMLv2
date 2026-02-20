package PowertrainSystem {
    private import ScalarValues::*;

    part def Powertrain;
    part def Engine;
    part def TorqueGenerator;

    attribute minTorque : Real = 0.0;

    port def TorquePort {
        out torque : Real;
    }

    part def Vehicle {
        part engine : Engine;
        part torqueGenerator : TorqueGenerator;
        port torqueOut : TorquePort;
    }

    part vehicleInstance : Vehicle {
        part engineInstance : Engine;
        part torqueGeneratorInstance : TorqueGenerator {
            attribute generatedTorque : Real;
            port torqueOutput : TorquePort;
        }
        port torqueOutPort : TorquePort;
    }

    requirement TorqueGeneration {
        attribute requiredTorque : Real;
        attribute description : String = "The system shall generate torque greater than zero newton-meters during operation.";
    }

    requirement TorqueAllocation {
        attribute description : String = "The torque generator function shall be allocated to physical powertrain components.";
    }

    part powertrain : Powertrain {
        part engine : Engine;
        part torqueGenerator : TorqueGenerator {
            attribute generatedTorque : Real;
            port torqueOutput : TorquePort;
        }
    }

    connect vehicleInstance.torqueGeneratorInstance.torqueOutput to vehicleInstance.torqueOutPort;
}