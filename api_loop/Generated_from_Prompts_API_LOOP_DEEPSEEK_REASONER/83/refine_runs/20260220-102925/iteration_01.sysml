package PowertrainSystem {
    private import ScalarValues::*;

    part def LogicalTorqueGenerator {
        attribute inputPower : Real;
        attribute outputTorque : Real;
    }

    part def PowerUnit {
        attribute outputPower : Real;
        attribute outputTorque : Real;
    }

    part def Powertrain {
        part logicalTG : LogicalTorqueGenerator;
        part physicalPU : PowerUnit;
        attribute mapping : Boolean;
    }
}