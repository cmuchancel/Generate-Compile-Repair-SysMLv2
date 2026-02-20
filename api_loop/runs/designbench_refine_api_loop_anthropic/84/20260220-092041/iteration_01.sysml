package PowerTransmissionSystem {
    private import ScalarValues::*;

    part def PowerTransmissionSystem {
        attribute providePower : Boolean;
        attribute generateTorque : Boolean;
    }

    part def LogicalLayer {
        attribute deliverPower : Boolean;
        attribute torqueGenerationCapability : Boolean;
    }

    part def TorqueGenerator {
        attribute generatingTorque : Boolean;
    }

    part def PhysicalLayer {
        part powertrain : Powertrain;
    }

    part def Powertrain {
        part engine : Engine;
    }

    part def Engine {
        attribute generatingTorque : Boolean;
    }

    part powerTransmission : PowerTransmissionSystem {
        attribute providePower : Boolean = true;
        attribute generateTorque : Boolean = true;
    }

    part logicalLayer : LogicalLayer {
        attribute deliverPower : Boolean = true;
        attribute torqueGenerationCapability : Boolean = true;
        part torqueGenerator : TorqueGenerator {
            attribute generatingTorque : Boolean = true;
        }
    }

    part physicalLayer : PhysicalLayer {
        part powertrain : Powertrain {
            part engine : Engine {
                attribute generatingTorque : Boolean = true;
            }
        }
    }
}